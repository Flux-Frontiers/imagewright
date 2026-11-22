"""Shared fixtures: a clean environment, a real contract server, and fake model stacks.

Only true external boundaries are faked: the Flux2Klein model object, the
OpenAI client and the MLX cache. HTTP runs against a real server on an
ephemeral port.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.machinery import ModuleSpec
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import uvicorn
from PIL import Image

import imagewright.discovery

_ENV_VARS = (
    "IMAGE_BACKEND",
    "IMAGE_ENDPOINT",
    "IMAGE_MODEL",
    "IMAGE_API_KEY",
    "IMAGE_STEPS",
    "GUTENKG_IMAGE_ENDPOINT",
    "GUTENKG_IMAGE_MODEL",
    "OPENAI_API_KEY",
    "IMAGE_OUTPUT_DIR",
    "IMAGE_PRELOAD",
    "IMAGE_IDLE_UNLOAD_S",
    "MFLUX_SERVER_HOST",
    "MFLUX_SERVER_PORT",
    "SDXL_MODEL",
    "SDXL_BASE",
    "SDXL_OFFLINE",
    "SDXL_SERVER_HOST",
    "SDXL_SERVER_PORT",
    "MPS_DTYPE",
)


def closed_port_url() -> str:
    """Return an http URL on a localhost port that nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def hermetic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip image env vars and point the default probe list at closed ports.

    A developer's shell or a real server on localhost:8090 must not change a
    result, and no server writes outside the test's own directory.
    """
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IMAGE_OUTPUT_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(
        imagewright.discovery, "DEFAULT_IMAGE_ENDPOINTS", (closed_port_url(), closed_port_url())
    )


def png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    """Return a small RGB PNG."""
    buf = BytesIO()
    Image.new("RGB", size, color=(100, 150, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_png_b64() -> str:
    """Base64-encoded 4x4 RGB PNG, valid for PIL.Image.open."""
    return base64.b64encode(png_bytes()).decode()


# ---------------------------------------------------------------------------
# A real HTTP server speaking the image contract
# ---------------------------------------------------------------------------


@dataclass
class ContractServer:
    """State of an in-test image server.

    :param url: Base URL the server listens on.
    :param health_status: Status code ``/health`` answers with (404 = an older server).
    :param models_status: Status code ``/v1/models`` answers with.
    :param generate_status: Status code ``/v1/images/generations`` answers with.
    :param requests: ``(method, path, json body or None)`` for every request received.
    """

    url: str = ""
    health_status: int = 200
    models_status: int = 200
    generate_status: int = 200
    requests: list[tuple[str, str, Any]] = field(default_factory=list)

    @property
    def generate_payloads(self) -> list[dict[str, Any]]:
        """JSON bodies of the generation requests received."""
        return [body for method, path, body in self.requests if method == "POST"]


@pytest.fixture
def contract_server() -> Iterator[ContractServer]:
    """Run a threaded HTTP server answering ``/health``, ``/v1/models`` and generations."""
    state = ContractServer()
    png = base64.b64encode(png_bytes()).decode()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            state.requests.append(("GET", self.path, None))
            if self.path == "/health":
                self._reply(
                    state.health_status,
                    {
                        "status": "ok",
                        "backend": "flux",
                        "model": "flux2-klein-4b",
                        "loaded": False,
                        "idle_unload_s": 0,
                    },
                )
            elif self.path == "/v1/models":
                self._reply(state.models_status, {"data": [{"id": "flux2-klein-4b"}]})
            elif self.path == "/image.png":  # a hosted image, as an OpenAI URL reply points to
                data = png_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._reply(404, {"detail": "Not Found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            state.requests.append(("POST", self.path, body))
            if self.path != "/v1/images/generations":
                self._reply(404, {"detail": "Not Found"})
            elif state.generate_status != 200:
                self._reply(state.generate_status, {"detail": "render failed"})
            else:
                self._reply(200, {"created": 0, "data": [{"b64_json": png}]})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Fake model stacks (installed into sys.modules; never the real packages)
# ---------------------------------------------------------------------------


def _install_modules(monkeypatch: pytest.MonkeyPatch, dotted: str) -> ModuleType:
    """Put a fake module and each of its parent packages into ``sys.modules``.

    :returns: The innermost module.
    """
    parts = dotted.split(".")
    module = ModuleType(dotted)
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        module = ModuleType(name)
        module.__spec__ = ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, module)
    return module


@dataclass
class FakeFluxStack:
    """Records what the fake Flux2Klein and MLX saw.

    :param loads: ``model_path`` of every Flux2Klein constructed.
    :param calls: Keyword arguments of every ``generate_image`` call.
    :param cache_clears: How many times ``mlx.core.clear_cache`` ran.
    :param gate: When set, ``generate_image`` signals :attr:`rendering` and then
        waits for *gate*, so a test can act while a render is in flight.
    :param rendering: Set when a render starts.
    """

    loads: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    cache_clears: int = 0
    gate: threading.Event | None = None
    rendering: threading.Event = field(default_factory=threading.Event)


@pytest.fixture
def fake_flux(monkeypatch: pytest.MonkeyPatch) -> FakeFluxStack:
    """Install fake ``mflux...flux2_klein`` and ``mlx.core`` modules."""
    stack = FakeFluxStack()

    class Flux2Klein:
        def __init__(self, model_path: str) -> None:
            stack.loads.append(model_path)
            self._thread = threading.get_ident()

        def generate_image(self, **kwargs: Any) -> SimpleNamespace:
            # MLX ties a model to the thread that loaded it; the real one fails
            # this way when a render runs on another thread.
            if threading.get_ident() != self._thread:
                raise RuntimeError("There is no Stream(cpu, 0) in current thread.")
            stack.calls.append(kwargs)
            stack.rendering.set()
            if stack.gate is not None:
                assert stack.gate.wait(10), "test never opened the gate"
            return SimpleNamespace(image=Image.new("RGB", (kwargs["width"], kwargs["height"])))

    def clear_cache() -> None:
        stack.cache_clears += 1

    klein = _install_modules(monkeypatch, "mflux.models.flux2.variants.txt2img.flux2_klein")
    klein.Flux2Klein = Flux2Klein  # type: ignore[attr-defined]
    mlx_core = _install_modules(monkeypatch, "mlx.core")
    mlx_core.clear_cache = clear_cache  # type: ignore[attr-defined]
    return stack


@dataclass
class FakeOpenAI:
    """Records what the fake OpenAI client saw and controls what it returns.

    :param api_keys: ``api_key`` of every client constructed.
    :param calls: Keyword arguments of every ``images.generate`` call.
    :param b64_json: What the response item carries in ``b64_json``.
    :param url: What the response item carries in ``url``.
    """

    api_keys: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    b64_json: str | None = None
    url: str | None = None


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch, tiny_png_b64: str) -> FakeOpenAI:
    """Install a fake ``openai`` module whose client returns a tiny PNG."""
    state = FakeOpenAI(b64_json=tiny_png_b64)

    class Images:
        def generate(self, **kwargs: Any) -> SimpleNamespace:
            state.calls.append(kwargs)
            item = SimpleNamespace(b64_json=state.b64_json, url=state.url)
            return SimpleNamespace(data=[item])

    class OpenAI:
        def __init__(self, api_key: str) -> None:
            state.api_keys.append(api_key)
            self.images = Images()

    module = _install_modules(monkeypatch, "openai")
    module.OpenAI = OpenAI  # type: ignore[attr-defined]
    return state


@pytest.fixture
def no_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make mflux, mlx and openai unimportable, whether or not they are installed."""
    for name in ("mflux", "mlx", "mlx.core", "openai"):
        monkeypatch.setitem(sys.modules, name, None)


@dataclass
class FakeSdxlStack:
    """Controls the fake torch and diffusers stack and records what it saw.

    :param cuda: What ``torch.cuda.is_available()`` returns.
    :param mps: What ``torch.backends.mps.is_available()`` returns.
    :param pretrained: ``(class name, repo, kwargs)`` of every ``from_pretrained`` call.
    :param downloads: ``(repo, filename, kwargs)`` of every ``hf_hub_download`` call.
    :param scheduler_kwargs: Keyword arguments of every scheduler ``from_config`` call.
    :param calls: Keyword arguments of every pipeline call.
    :param seeds: Every seed given to a ``torch.Generator``, with its device.
    :param empty_cache: Devices whose cache was emptied, in order.
    :param unet_mismatch: ``(missing, unexpected)`` the UNet's ``load_state_dict`` returns.
    """

    cuda: bool = False
    mps: bool = True
    pretrained: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    downloads: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    scheduler_kwargs: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[tuple[str, int]] = field(default_factory=list)
    empty_cache: list[str] = field(default_factory=list)
    unet_mismatch: tuple[list[str], list[str]] = (["a"], [])


@pytest.fixture
def fake_sdxl(monkeypatch: pytest.MonkeyPatch) -> FakeSdxlStack:
    """Install fake torch, diffusers, huggingface_hub, safetensors, transformers, accelerate."""
    stack = FakeSdxlStack()

    torch = _install_modules(monkeypatch, "torch")
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: stack.cuda,
        empty_cache=lambda: stack.empty_cache.append("cuda"),
    )
    torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: stack.mps)
    )
    torch.mps = SimpleNamespace(  # type: ignore[attr-defined]
        empty_cache=lambda: stack.empty_cache.append("mps")
    )

    class Generator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> Generator:
            stack.seeds.append((self.device, seed))
            return self

    torch.Generator = Generator  # type: ignore[attr-defined]

    class Part:
        """A VAE, UNet or scheduler: whatever the loader touches on it."""

        config = {"name": "scheduler"}

        def to(self, device: str) -> Part:
            self.device = device
            return self

        def enable_slicing(self) -> None:
            self.sliced = True

        def load_state_dict(self, state: Any, strict: bool) -> tuple[list[str], list[str]]:
            return stack.unet_mismatch

    class Pipeline(Part):
        def __init__(self) -> None:
            self.vae = Part()
            self.unet = Part()
            self.scheduler = Part()

        def enable_attention_slicing(self) -> None:
            self.attention_sliced = True

        def __call__(self, **kwargs: Any) -> SimpleNamespace:
            stack.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))])

    def pretrained(cls_name: str, factory: Any) -> Any:
        def from_pretrained(repo: str, **kwargs: Any) -> Any:
            stack.pretrained.append((cls_name, repo, kwargs))
            return factory()

        return SimpleNamespace(from_pretrained=from_pretrained)

    def from_config(config: Any, **kwargs: Any) -> Part:
        stack.scheduler_kwargs.append(kwargs)
        return Part()

    diffusers = _install_modules(monkeypatch, "diffusers")
    diffusers.StableDiffusionXLPipeline = pretrained("pipeline", Pipeline)  # type: ignore[attr-defined]
    diffusers.AutoencoderKL = pretrained("vae", Part)  # type: ignore[attr-defined]
    diffusers.EulerDiscreteScheduler = SimpleNamespace(from_config=from_config)  # type: ignore[attr-defined]

    def hf_hub_download(repo: str, filename: str, **kwargs: Any) -> str:
        stack.downloads.append((repo, filename, kwargs))
        return f"/cache/{filename}"

    _install_modules(monkeypatch, "huggingface_hub").hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    _install_modules(monkeypatch, "safetensors.torch").load_file = lambda path, device: {}  # type: ignore[attr-defined]
    _install_modules(monkeypatch, "transformers")
    _install_modules(monkeypatch, "accelerate")
    return stack


# ---------------------------------------------------------------------------
# A real uvicorn server on an ephemeral port
# ---------------------------------------------------------------------------


@contextmanager
def live_server(app: Any) -> Iterator[str]:
    """Serve *app* with uvicorn on an ephemeral 127.0.0.1 port, lifespan included.

    :returns: The base URL, for the life of the ``with`` block.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
