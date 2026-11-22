"""The FLUX server: the real FastAPI app around a real ImageSynthesizer and a fake Flux2Klein."""

import base64
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import imagewright.discovery
from imagewright import ImageBackend, ImageConfig, ImageSynthesizer
from imagewright.discovery import discover_image_endpoint, resolve_image_config
from imagewright.flux_server import FluxServer, create_app
from tests.conftest import FakeFluxStack, closed_port_url, live_server

MODEL = "mlx-community/flux2-klein-4b-4bit"


@pytest.fixture
def server(fake_flux: FakeFluxStack, tmp_path: Path) -> FluxServer:
    return FluxServer(output_dir=tmp_path / "out")


@pytest.fixture
def client(server: FluxServer) -> TestClient:
    return TestClient(create_app(server))


def decode(reply_json: dict) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(reply_json["data"][0]["b64_json"])))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_health_reports_the_contract_keys(client: TestClient):
    reply = client.get("/health")
    assert reply.status_code == 200
    assert reply.json() == {
        "status": "ok",
        "backend": "flux",
        "model": "flux2-klein-4b",
        "loaded": False,
        "idle_unload_s": 0,
    }


def test_health_reports_a_loaded_model_after_a_render(client: TestClient):
    client.post("/v1/images/generations", json={"prompt": "p"})
    assert client.get("/health").json()["loaded"] is True


def test_models_lists_the_one_served_model(client: TestClient):
    assert client.get("/v1/models").json() == {
        "object": "list",
        "data": [{"id": "flux2-klein-4b", "object": "model", "owned_by": "mflux"}],
    }


def test_generate_returns_base64_png(client: TestClient, fake_flux: FakeFluxStack):
    reply = client.post(
        "/v1/images/generations",
        json={"prompt": "a lighthouse", "size": "320x200", "seed": 7, "num_inference_steps": 3},
    )
    assert reply.status_code == 200
    body = reply.json()
    assert isinstance(body["created"], int)
    image = decode(body)
    assert image.format == "PNG"
    assert image.size == (320, 200)
    assert fake_flux.loads == [MODEL]
    call = fake_flux.calls[0]
    assert (call["prompt"], call["seed"], call["num_inference_steps"]) == ("a lighthouse", 7, 3)
    assert (call["width"], call["height"]) == (320, 200)


def test_generate_writes_a_file_for_filepath(
    client: TestClient, server: FluxServer, fake_flux: FakeFluxStack
):
    reply = client.post(
        "/v1/images/generations",
        json={"prompt": "p", "size": "64x48", "response_format": "filepath"},
    )
    assert reply.status_code == 200
    path = Path(reply.json()["data"][0]["filepath"])
    assert path.parent == server.output_dir
    assert path.suffix == ".png"
    assert Image.open(path).size == (64, 48)


def test_request_defaults_match_gutenbergs_server(client: TestClient, fake_flux: FakeFluxStack):
    image = decode(client.post("/v1/images/generations", json={"prompt": "p"}).json())
    assert image.size == (1536, 1024)
    call = fake_flux.calls[0]
    assert call["num_inference_steps"] == 4
    assert isinstance(call["seed"], int)


@pytest.mark.parametrize("steps", [None, 0])
def test_missing_or_zero_steps_use_the_server_default(
    fake_flux: FakeFluxStack, tmp_path: Path, steps: int | None
):
    client = TestClient(create_app(FluxServer(steps=6, output_dir=tmp_path)))
    client.post("/v1/images/generations", json={"prompt": "p", "num_inference_steps": steps})
    assert fake_flux.calls[0]["num_inference_steps"] == 6


def test_a_bad_size_falls_back_to_the_default(client: TestClient):
    reply = client.post("/v1/images/generations", json={"prompt": "p", "size": "huge"})
    assert decode(reply.json()).size == (1536, 1024)


def test_an_unknown_response_format_returns_base64(client: TestClient):
    reply = client.post("/v1/images/generations", json={"prompt": "p", "response_format": "url"})
    assert "b64_json" in reply.json()["data"][0]


def test_a_request_without_a_prompt_is_rejected(client: TestClient):
    assert client.post("/v1/images/generations", json={}).status_code == 422


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("IMAGE_OUTPUT_DIR")
    srv = FluxServer.from_env()
    assert srv.model_name == MODEL
    assert srv.steps == 4
    assert srv.preload is False
    assert srv.idle_unload_s == 0
    assert srv.output_dir.name == "imagewright_images"


def test_from_env_reads_every_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("IMAGE_MODEL", "my/model")
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "old/model")
    monkeypatch.setenv("IMAGE_STEPS", "8")
    monkeypatch.setenv("IMAGE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_PRELOAD", " Yes ")
    monkeypatch.setenv("IMAGE_IDLE_UNLOAD_S", "300")
    srv = FluxServer.from_env()
    assert (srv.model_name, srv.steps, srv.output_dir) == ("my/model", 8, tmp_path)
    assert (srv.preload, srv.idle_unload_s) == (True, 300)


def test_gutenkg_image_model_is_the_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "old/model")
    assert FluxServer.from_env().model_name == "old/model"


def test_create_app_reads_the_environment_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IMAGE_IDLE_UNLOAD_S", "42")
    app = create_app()
    assert TestClient(app).get("/health").json()["idle_unload_s"] == 42


# ---------------------------------------------------------------------------
# Preload, the render lock and idle unload
# ---------------------------------------------------------------------------


def test_preload_loads_before_the_first_request(fake_flux: FakeFluxStack, tmp_path: Path):
    app = create_app(FluxServer(preload=True, output_dir=tmp_path))
    with TestClient(app) as client:
        assert fake_flux.loads == [MODEL]
        assert client.get("/health").json()["loaded"] is True
        client.post("/v1/images/generations", json={"prompt": "p"})
    assert fake_flux.loads == [MODEL]  # the render reused the preloaded model


def test_without_preload_nothing_loads_at_startup(fake_flux: FakeFluxStack, tmp_path: Path):
    with TestClient(create_app(FluxServer(output_dir=tmp_path))) as client:
        assert client.get("/health").json()["loaded"] is False
    assert fake_flux.loads == []


@pytest.fixture
def loaded(fake_flux: FakeFluxStack, tmp_path: Path) -> FluxServer:
    srv = FluxServer(idle_unload_s=300, output_dir=tmp_path)
    srv.load()
    srv.last_used = 1000.0
    return srv


def test_unloads_after_the_idle_time(loaded: FluxServer, fake_flux: FakeFluxStack):
    assert loaded.maybe_unload(now=1300.0) is True
    assert not loaded.synth.model_loaded
    assert fake_flux.cache_clears == 1


def test_the_next_render_reloads(loaded: FluxServer, fake_flux: FakeFluxStack):
    loaded.maybe_unload(now=1300.0)
    loaded.render("p", "64x64", None, None)
    assert fake_flux.loads == [MODEL, MODEL]
    assert loaded.synth.model_loaded


def test_keeps_a_recently_used_model(loaded: FluxServer):
    assert loaded.maybe_unload(now=1299.0) is False
    assert loaded.synth.model_loaded


def test_idle_unload_is_off_by_default(loaded: FluxServer):
    loaded.idle_unload_s = 0
    assert loaded.maybe_unload(now=10**9) is False
    assert loaded.synth.model_loaded


def test_nothing_to_unload(server: FluxServer):
    server.idle_unload_s = 300
    assert server.maybe_unload(now=10**9) is False


def test_a_render_marks_the_model_used(server: FluxServer):
    server.last_used = 0.0
    server.render("p", "64x64", None, None)
    assert server.last_used > 0.0


def test_a_render_in_flight_blocks_an_unload(
    fake_flux: FakeFluxStack, loaded: FluxServer, tmp_path: Path
):
    fake_flux.gate = threading.Event()
    client = TestClient(create_app(loaded))
    request = threading.Thread(
        target=client.post, args=("/v1/images/generations",), kwargs={"json": {"prompt": "p"}}
    )
    request.start()
    try:
        assert fake_flux.rendering.wait(10)
        assert loaded.maybe_unload(now=10**9) is False
        assert loaded.synth.model_loaded
    finally:
        fake_flux.gate.set()
        request.join(10)
    assert loaded.maybe_unload(now=10**9) is True


def test_the_watcher_unloads_an_idle_model(fake_flux: FakeFluxStack, tmp_path: Path):
    srv = FluxServer(idle_unload_s=1, output_dir=tmp_path)
    with TestClient(create_app(srv, watch_interval=0.01)) as client:
        client.post("/v1/images/generations", json={"prompt": "p"})
        srv.last_used = time.monotonic() - 10
        deadline = time.monotonic() + 5
        while client.get("/health").json()["loaded"]:
            assert time.monotonic() < deadline, "the watcher never unloaded the model"
            time.sleep(0.01)
    assert not any(t.name == "flux-idle-unload" for t in threading.enumerate())


def test_watch_idle_returns_once_stopped(server: FluxServer):
    stop = threading.Event()
    stop.set()
    server.idle_unload_s = 4
    server.watch_idle(stop)  # the default interval (5 s) is never waited out


# ---------------------------------------------------------------------------
# The real client and discovery against the real app
# ---------------------------------------------------------------------------


def test_client_and_discovery_agree_with_the_server(
    server: FluxServer, fake_flux: FakeFluxStack, monkeypatch: pytest.MonkeyPatch
):
    with live_server(create_app(server)) as url:
        assert discover_image_endpoint((closed_port_url(), url)) == url

        monkeypatch.setattr(imagewright.discovery, "DEFAULT_IMAGE_ENDPOINTS", (url,))
        config = resolve_image_config()
        assert (config.backend, config.server_url) == (ImageBackend.MFLUX_SERVE, url)

        image = ImageSynthesizer(config).generate("a lighthouse", size="320x200", seed=3, steps=2)
    assert image.size == (320, 200)
    call = fake_flux.calls[0]
    assert (call["prompt"], call["seed"], call["num_inference_steps"]) == ("a lighthouse", 3, 2)


def test_generate_b64_through_the_server(server: FluxServer):
    with live_server(create_app(server)) as url:
        config = ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url=url)
        b64 = ImageSynthesizer(config).generate_b64("p", aspect_ratio="1:1")
    assert Image.open(BytesIO(base64.b64decode(b64))).size == (1024, 1024)


# ---------------------------------------------------------------------------
# A core install
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", ["imagewright.flux_server", "imagewright.sdxl_server", "imagewright.mcp_server"]
)
def test_importing_a_server_loads_no_server_or_model_stack(module: str):
    heavy = (
        "fastapi",
        "uvicorn",
        "pydantic",
        "starlette",
        "mcp",
        "mflux",
        "mlx",
        "torch",
        "diffusers",
    )
    code = (
        f"import sys, {module}\n"
        f"loaded = [m for m in {heavy!r} if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
