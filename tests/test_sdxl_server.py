"""The SDXL server: the real FastAPI app over a fake torch and diffusers stack.

Ported from gutenberg_kg's ``tests/test_sdxl_server.py``. The fakes stand in
for the model stack only; routing, size and step handling, the pipeline
assembly and the response bodies are the real code.
"""

import base64
import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagewright import MissingExtraError
from imagewright.sdxl_server import (
    DEFAULT_DIMS,
    LIGHTNING_REPO,
    VAE_FP16_FIX,
    SdxlServer,
    create_app,
    device_dtype,
    parse_size,
    steps_for,
)
from tests.conftest import FakeSdxlStack, live_server

# ---------------------------------------------------------------------------
# Size, steps, device
# ---------------------------------------------------------------------------


class TestParseSize:
    def test_parses_plain_size(self):
        assert parse_size("768x512") == (768, 512)

    def test_parses_uppercase_x(self):
        assert parse_size("1536X1024") == (1536, 1024)

    @pytest.mark.parametrize("bad", [None, "", "1024", "axb", "1024x", "x1024", "1024x1024x1024"])
    def test_malformed_falls_back_to_default(self, bad):
        assert parse_size(bad) == DEFAULT_DIMS == (1024, 1024)

    @pytest.mark.parametrize("bad", ["0x512", "512x0", "-1x512"])
    def test_non_positive_falls_back_to_default(self, bad):
        assert parse_size(bad) == DEFAULT_DIMS


class TestStepsFor:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [("sdxl_lightning_2", 2), ("sdxl_lightning_4", 4), ("sdxl_lightning_8", 8)],
    )
    def test_model_name_determines_steps(self, model, expected):
        assert steps_for(model) == expected

    def test_model_wins_over_a_request_override(self):
        assert steps_for("sdxl_lightning_4", 30) == 4

    def test_unknown_model_honors_the_request(self):
        assert steps_for("something-else", 6) == 6

    @pytest.mark.parametrize("requested", [None, 0, -1])
    def test_unknown_model_without_a_usable_request_defaults_to_four(self, requested):
        assert steps_for("something-else", requested) == 4


class TestDeviceDtype:
    def test_cuda_preferred(self, fake_sdxl: FakeSdxlStack):
        fake_sdxl.cuda = True
        assert device_dtype() == ("cuda", "float16")

    def test_mps_when_no_cuda(self, fake_sdxl: FakeSdxlStack):
        assert device_dtype() == ("mps", "float16")

    def test_mps_dtype_override(self, fake_sdxl: FakeSdxlStack, monkeypatch):
        monkeypatch.setenv("MPS_DTYPE", "float32")
        assert device_dtype() == ("mps", "float32")

    def test_cpu_fallback(self, fake_sdxl: FakeSdxlStack):
        fake_sdxl.mps = False
        assert device_dtype() == ("cpu", "float32")

    def test_missing_mps_backend_is_survivable(self, fake_sdxl: FakeSdxlStack):
        sys.modules["torch"].backends = object()  # no `mps` attribute at all
        assert device_dtype() == ("cpu", "float32")

    def test_missing_torch_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(MissingExtraError, match=r'pip install "imagewright\[sdxl\]"'):
            device_dtype()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("IMAGE_OUTPUT_DIR")
    srv = SdxlServer.from_env()
    assert (srv.model, srv.base, srv.offline) == (
        "sdxl_lightning_4",
        "stabilityai/stable-diffusion-xl-base-1.0",
        False,
    )
    assert srv.output_dir.name == "imagewright_images"


def test_from_env_reads_every_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SDXL_MODEL", "sdxl_lightning_8")
    monkeypatch.setenv("SDXL_BASE", "my/base")
    monkeypatch.setenv("SDXL_OFFLINE", "TRUE")
    monkeypatch.setenv("IMAGE_OUTPUT_DIR", str(tmp_path))
    srv = SdxlServer.from_env()
    assert (srv.model, srv.base, srv.offline, srv.output_dir) == (
        "sdxl_lightning_8",
        "my/base",
        True,
        tmp_path,
    )


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_other_offline_values_leave_downloads_enabled(monkeypatch, value):
    monkeypatch.setenv("SDXL_OFFLINE", value)
    assert SdxlServer.from_env().offline is False


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def test_load_pipeline_on_mps_swaps_in_the_fp16_vae(fake_sdxl: FakeSdxlStack, tmp_path: Path):
    srv = SdxlServer(model="sdxl_lightning_2", output_dir=tmp_path)
    pipe = srv.load_pipeline()
    assert fake_sdxl.pretrained == [
        (
            "pipeline",
            "stabilityai/stable-diffusion-xl-base-1.0",
            {"torch_dtype": "float16", "local_files_only": False},
        ),
        ("vae", VAE_FP16_FIX, {"torch_dtype": "float16", "local_files_only": False}),
    ]
    assert fake_sdxl.downloads == [
        (LIGHTNING_REPO, "sdxl_lightning_2step_unet.safetensors", {"local_files_only": False})
    ]
    assert fake_sdxl.scheduler_kwargs == [{"timestep_spacing": "trailing"}]
    assert pipe.device == pipe.vae.device == "mps"
    assert pipe.attention_sliced and pipe.vae.sliced
    assert srv.load_pipeline() is pipe  # cached
    assert len(fake_sdxl.pretrained) == 2


def test_load_pipeline_on_cpu_keeps_the_stock_vae_and_honors_offline(
    fake_sdxl: FakeSdxlStack, tmp_path: Path
):
    fake_sdxl.mps = False
    fake_sdxl.unet_mismatch = ([], [])
    SdxlServer(offline=True, output_dir=tmp_path).load_pipeline()
    assert [(name, kw) for name, _, kw in fake_sdxl.pretrained] == [
        ("pipeline", {"torch_dtype": "float32", "local_files_only": True})
    ]
    assert fake_sdxl.downloads[0][1:] == (
        "sdxl_lightning_4step_unet.safetensors",
        {"local_files_only": True},
    )


def test_load_pipeline_without_diffusers_names_the_extra(fake_sdxl, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(MissingExtraError, match=r"imagewright\[sdxl\]"):
        SdxlServer(output_dir=tmp_path).load_pipeline()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def server(fake_sdxl: FakeSdxlStack, tmp_path: Path) -> SdxlServer:
    return SdxlServer(output_dir=tmp_path / "out")


@pytest.fixture
def client(server: SdxlServer) -> TestClient:
    return TestClient(create_app(server))


def test_health_answers_without_touching_the_model(monkeypatch, tmp_path: Path):
    # No fake stack: torch is not importable, so any model work would fail.
    monkeypatch.setitem(sys.modules, "torch", None)
    reply = TestClient(create_app(SdxlServer(output_dir=tmp_path))).get("/health")
    assert reply.json() == {
        "status": "ok",
        "backend": "sdxl-lightning",
        "model": "sdxl_lightning_4",
    }


def test_models_reports_the_active_variant(fake_sdxl, tmp_path: Path):
    client = TestClient(create_app(SdxlServer(model="sdxl_lightning_8", output_dir=tmp_path)))
    assert client.get("/v1/models").json() == {
        "object": "list",
        "data": [{"id": "sdxl_lightning_8", "object": "model", "owned_by": "sdxl-lightning"}],
    }


def test_request_defaults(client: TestClient, fake_sdxl: FakeSdxlStack):
    reply = client.post("/v1/images/generations", json={"prompt": "a candlelit study"})
    image = Image.open(BytesIO(base64.b64decode(reply.json()["data"][0]["b64_json"])))
    assert image.size == (1024, 1024)
    call = fake_sdxl.calls[0]
    assert call["negative_prompt"] == "blurry, bad quality, distorted"
    assert (call["num_inference_steps"], call["guidance_scale"]) == (4, 1.0)
    assert call["generator"] is None
    assert fake_sdxl.empty_cache == ["mps"]


def test_size_seed_and_steps_reach_the_pipeline(client: TestClient, fake_sdxl: FakeSdxlStack):
    client.post(
        "/v1/images/generations",
        json={
            "prompt": "p",
            "size": "768x512",
            "seed": 11,
            "num_inference_steps": 30,
            "negative_prompt": "fog",
        },
    )
    call = fake_sdxl.calls[0]
    assert (call["width"], call["height"]) == (768, 512)
    assert call["num_inference_steps"] == 4  # the 4-step UNet ignores the request
    assert call["negative_prompt"] == "fog"
    assert fake_sdxl.seeds == [("mps", 11)]


def test_an_unknown_variant_honors_requested_steps(fake_sdxl, tmp_path: Path):
    client = TestClient(create_app(SdxlServer(model="custom", output_dir=tmp_path)))
    client.post("/v1/images/generations", json={"prompt": "p", "num_inference_steps": 6})
    assert fake_sdxl.calls[0]["num_inference_steps"] == 6


def test_a_bad_size_falls_back_to_1024(client: TestClient, fake_sdxl: FakeSdxlStack):
    client.post("/v1/images/generations", json={"prompt": "p", "size": "big"})
    assert (fake_sdxl.calls[0]["width"], fake_sdxl.calls[0]["height"]) == (1024, 1024)


def test_cuda_empties_its_cache(client: TestClient, fake_sdxl: FakeSdxlStack):
    fake_sdxl.cuda = True
    client.post("/v1/images/generations", json={"prompt": "p"})
    assert fake_sdxl.empty_cache == ["cuda"]


def test_filepath_writes_a_png(client: TestClient, server: SdxlServer):
    reply = client.post(
        "/v1/images/generations",
        json={"prompt": "p", "size": "64x32", "response_format": "filepath"},
    )
    path = Path(reply.json()["data"][0]["filepath"])
    assert path.parent == server.output_dir
    assert Image.open(path).size == (64, 32)


def test_startup_builds_the_pipeline_before_serving(server: SdxlServer, fake_sdxl):
    with live_server(create_app(server)):
        assert server.pipe is not None
        assert len(fake_sdxl.pretrained) == 2


def test_create_app_reads_the_environment_by_default(fake_sdxl, monkeypatch):
    monkeypatch.setenv("SDXL_MODEL", "sdxl_lightning_2")
    assert TestClient(create_app()).get("/health").json()["model"] == "sdxl_lightning_2"
