"""FLUX.2 Klein image server: the ``/v1/images/generations`` contract over an in-process model.

A FastAPI app around one long-lived :class:`~imagewright.client.ImageSynthesizer`
in ``mflux-local`` mode, so the model stays loaded between requests. It
answers the same routes, request fields and response bodies as gutenberg_kg's
``serve/image_server.py``, so a client of that server can point here unchanged.

Run it with ``imagewright serve``. It needs the ``serve`` and ``flux`` extras;
nothing at module scope imports either, so this module imports on a core install.

Environment variables (read by :meth:`FluxServer.from_env`)
-----------------------------------------------------------
IMAGE_MODEL           HF repo for Flux2Klein. Fallback: GUTENKG_IMAGE_MODEL.
                      Default: mlx-community/flux2-klein-4b-4bit
IMAGE_STEPS           Default inference steps (default: 4)
IMAGE_OUTPUT_DIR      Where response_format=filepath images go
                      (default: imagewright_images in the system temp directory)
IMAGE_PRELOAD         1/true/yes loads the model before the server accepts requests
IMAGE_IDLE_UNLOAD_S   Unload FLUX after this many idle seconds, so a small Mac gets
                      its 12-15 GB back; the next render reloads it. 0 (default)
                      keeps it loaded.
"""

import asyncio
import os
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from PIL.Image import Image as PILImage

from imagewright._serving import SERVE_MODULES, env_flag, env_output_dir, image_reply
from imagewright._serving import require_modules as _require_modules
from imagewright.client import ImageSynthesizer
from imagewright.config import _IMAGE_DEFAULTS, ImageBackend, ImageConfig

#: Port gutenberg_kg's FLUX server has always used; discovery probes it first.
DEFAULT_PORT = 8090

#: The model id ``/health``, ``/v1/models`` and the request default report. Fixed by
#: the contract, whatever HF repo is loaded.
SERVED_MODEL_ID = "flux2-klein-4b"

_DEFAULT_MODEL = _IMAGE_DEFAULTS[ImageBackend.MFLUX_LOCAL]["model"]


def require_extras() -> None:
    """Check that the FLUX server can run here.

    :raises MissingExtraError: Naming ``pip install "imagewright[serve,flux]"``.
    """
    _require_modules((*SERVE_MODULES, "mflux"), "serve,flux", "the FLUX server")


class FluxServer:
    """The model, the render lock and the idle clock behind the FLUX app.

    One render runs at a time, and the model is never unloaded in the middle of
    one: :meth:`render` and :meth:`maybe_unload` share one lock.

    :param model_name: HF repo for Flux2Klein.
    :param steps: Inference steps when a request does not set them.
    :param output_dir: Where ``response_format=filepath`` images go.
    :param preload: Load the model when the app starts, before it accepts requests.
    :param idle_unload_s: Unload the model after this many idle seconds; 0 never does.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        steps: int = 4,
        output_dir: Path | None = None,
        preload: bool = False,
        idle_unload_s: int = 0,
    ) -> None:
        self.model_name = model_name
        self.steps = steps
        self.output_dir = output_dir or env_output_dir()
        self.preload = preload
        self.idle_unload_s = idle_unload_s
        self.synth = ImageSynthesizer(
            ImageConfig(backend=ImageBackend.MFLUX_LOCAL, model=model_name, steps=steps)
        )
        self.lock = threading.Lock()
        self.last_used = time.monotonic()

    @classmethod
    def from_env(cls) -> "FluxServer":
        """Build a server from the environment variables in the module docstring.

        :returns: The configured server.
        """
        return cls(
            model_name=os.environ.get("IMAGE_MODEL")
            or os.environ.get("GUTENKG_IMAGE_MODEL")
            or _DEFAULT_MODEL,
            steps=int(os.environ.get("IMAGE_STEPS") or 4),
            output_dir=env_output_dir(),
            preload=env_flag("IMAGE_PRELOAD"),
            idle_unload_s=int(os.environ.get("IMAGE_IDLE_UNLOAD_S") or 0),
        )

    def load(self) -> None:
        """Load the model now, under the render lock."""
        with self.lock:
            self.synth.load_model()

    def render(self, prompt: str, size: str, seed: int | None, steps: int | None) -> PILImage:
        """Render one image under the lock and note when it finished.

        :param prompt: Text description of the image.
        :param size: ``"WIDTHxHEIGHT"``; one that does not parse gives 1536x1024.
        :param seed: Random seed, or None for a random one.
        :param steps: Inference steps; None or 0 uses the server default.
        :returns: The rendered image.
        """
        with self.lock:
            try:
                return self.synth.generate(prompt, size=size, seed=seed, steps=steps or self.steps)
            finally:
                self.last_used = time.monotonic()

    def maybe_unload(self, now: float | None = None) -> bool:
        """Unload the model if it has been idle for ``idle_unload_s`` seconds.

        Skipped while a render holds the lock; the watcher tries again later.

        :param now: The current ``time.monotonic()`` reading (for tests).
        :returns: ``True`` if the model was unloaded.
        """
        if self.idle_unload_s <= 0 or not self.synth.model_loaded:
            return False
        now = time.monotonic() if now is None else now
        if now - self.last_used < self.idle_unload_s:
            return False
        if not self.lock.acquire(blocking=False):
            return False
        try:
            return self.synth.unload_model()
        finally:
            self.lock.release()

    def watch_idle(self, stop: threading.Event, interval: float | None = None) -> None:
        """Check for an idle model every few seconds until *stop* is set.

        :param stop: Set it to end the loop.
        :param interval: Seconds between checks. Default: a quarter of
            ``idle_unload_s``, kept between 5 and 30.
        """
        if interval is None:
            interval = max(5.0, min(30.0, self.idle_unload_s / 4))
        while not stop.wait(interval):
            if self.maybe_unload():
                print(
                    f"[idle] FLUX unloaded after {self.idle_unload_s}s idle; "
                    "the next render reloads it",
                    flush=True,
                )


def create_app(server: FluxServer | None = None, *, watch_interval: float | None = None) -> Any:
    """Build the FastAPI app.

    On startup the app loads the model when ``server.preload`` is set, and starts
    the idle watcher when ``server.idle_unload_s`` is above 0; shutdown stops it.
    ``app.state.server`` is *server*.

    :param server: The model and its settings. Default: :meth:`FluxServer.from_env`.
    :param watch_interval: Seconds between idle checks (for tests); see
        :meth:`FluxServer.watch_idle`.
    :returns: A ``fastapi.FastAPI`` app.
    """
    from fastapi import FastAPI  # noqa: PLC0415 - the `serve` extra
    from pydantic import BaseModel  # noqa: PLC0415 - the `serve` extra

    srv = server or FluxServer.from_env()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        if srv.preload:
            print(f"[startup] loading model {srv.model_name} ...", flush=True)
            srv.load()
            print("[startup] model ready", flush=True)
        else:
            print(
                "[startup] IMAGE_PRELOAD disabled; model will load on first generation request",
                flush=True,
            )
        stop = threading.Event()
        watcher = None
        if srv.idle_unload_s > 0:
            watcher = threading.Thread(
                target=srv.watch_idle,
                args=(stop, watch_interval),
                name="flux-idle-unload",
                daemon=True,
            )
            watcher.start()
            print(
                f"[startup] FLUX unloads after {srv.idle_unload_s}s idle (IMAGE_IDLE_UNLOAD_S)",
                flush=True,
            )
        try:
            yield
        finally:
            stop.set()
            if watcher is not None:
                watcher.join()

    app = FastAPI(title="imagewright FLUX image server", lifespan=lifespan)
    app.state.server = srv

    class ImageGenRequest(BaseModel):
        model: str = SERVED_MODEL_ID
        prompt: str
        n: int = 1
        size: str = "1536x1024"
        quality: str | None = None
        num_inference_steps: int | None = None
        seed: int | None = None
        response_format: str = "b64_json"

    @app.get("/health")
    def health():
        """Liveness probe.

        Deliberately does no model work: it answers while a generation is in
        flight and before any weights are resident, so a caller can tell "this
        port is mine" apart from "this port is ready to be fast".
        """
        return {
            "status": "ok",
            "backend": "flux",
            "model": SERVED_MODEL_ID,
            "loaded": srv.synth.model_loaded,
            "idle_unload_s": srv.idle_unload_s,
        }

    @app.get("/v1/models")
    def list_models():
        """List the single model this server serves, in OpenAI-compatible format."""
        return {
            "object": "list",
            "data": [{"id": SERVED_MODEL_ID, "object": "model", "owned_by": "mflux"}],
        }

    @app.post("/v1/images/generations")
    async def generate_image(req: ImageGenRequest):
        """Render in a worker thread and return the image as base64 or a file path."""
        image = await asyncio.to_thread(
            srv.render, req.prompt, req.size, req.seed, req.num_inference_steps
        )
        return image_reply(image, req.response_format, srv.output_dir)

    return app
