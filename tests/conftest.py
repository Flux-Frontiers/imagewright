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
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.machinery import ModuleSpec
from io import BytesIO
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
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
)


def closed_port_url() -> str:
    """Return an http URL on a localhost port that nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip image env vars and point the default probe list at closed ports.

    A developer's shell or a real server on localhost:8090 must not change a result.
    """
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
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
    """

    loads: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    cache_clears: int = 0


@pytest.fixture
def fake_flux(monkeypatch: pytest.MonkeyPatch) -> FakeFluxStack:
    """Install fake ``mflux...flux2_klein`` and ``mlx.core`` modules."""
    stack = FakeFluxStack()

    class Flux2Klein:
        def __init__(self, model_path: str) -> None:
            stack.loads.append(model_path)

        def generate_image(self, **kwargs: Any) -> SimpleNamespace:
            stack.calls.append(kwargs)
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
