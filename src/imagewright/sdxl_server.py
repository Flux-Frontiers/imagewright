"""SDXL-Lightning image server: the ``/v1/images/generations`` contract over diffusers.

A lighter alternative to :mod:`imagewright.flux_server` that runs on CUDA, MPS
or CPU: SDXL base with a ByteDance SDXL-Lightning UNet (2, 4 or 8 steps). It
answers the same routes, request fields and response bodies as gutenberg_kg's
``serve/sdxl_server.py``.

Run it with ``imagewright serve --backend sdxl``. It needs the ``serve`` and
``sdxl`` extras. torch and diffusers are imported only inside the functions
that need them, so this module imports on a core install.

Environment variables
---------------------
``SdxlServer.from_env`` reads these.

SDXL_MODEL        Lightning variant: sdxl_lightning_2 | _4 (default) | _8
SDXL_BASE         SDXL base repo (default: stabilityai/stable-diffusion-xl-base-1.0)
SDXL_OFFLINE      1 = never reach HuggingFace; fail if weights are not cached
MPS_DTYPE         float16 (default; uses the fp16-fix VAE) | float32 (heavier)
IMAGE_OUTPUT_DIR  Where response_format=filepath images go
                  (default: imagewright_images in the system temp directory)
"""

import asyncio
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from PIL.Image import Image as PILImage

from imagewright._serving import SERVE_MODULES, env_flag, env_output_dir, image_reply
from imagewright._serving import require_modules as _require_modules
from imagewright.client import _import_extra

#: Port gutenberg_kg's SDXL server has always used; discovery probes it second.
DEFAULT_PORT = 8091

DEFAULT_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
#: fp16-safe VAE: the SDXL VAE overflows in float16 and yields black images; this
#: fixed VAE lets the whole pipeline run float16 on MPS (about half the memory).
VAE_FP16_FIX = "madebyollin/sdxl-vae-fp16-fix"

#: SDXL-Lightning enforces its trained step count; guidance is about 1.0 (no CFG).
LIGHTNING_STEPS = {"sdxl_lightning_2": 2, "sdxl_lightning_4": 4, "sdxl_lightning_8": 8}

DEFAULT_DIMS: tuple[int, int] = (1024, 1024)

_SDXL_MODULES = ("torch", "diffusers", "transformers", "accelerate", "safetensors")


def require_extras() -> None:
    """Check that the SDXL server can run here.

    :raises MissingExtraError: Naming ``pip install "imagewright[serve,sdxl]"``.
    """
    _require_modules(
        (*SERVE_MODULES, *_SDXL_MODULES, "huggingface_hub"), "serve,sdxl", "the SDXL server"
    )


def _torch() -> Any:
    """Import torch on demand.

    :returns: The :mod:`torch` module.
    :raises MissingExtraError: Naming ``pip install "imagewright[sdxl]"``.
    """
    return _import_extra("torch", "sdxl")


def parse_size(size: str | None) -> tuple[int, int]:
    """Parse ``"WIDTHxHEIGHT"`` into a ``(width, height)`` pair.

    :param size: Size string such as ``"1024x1024"`` (case-insensitive ``x``).
    :returns: The parsed pair, or :data:`DEFAULT_DIMS` when it does not parse to
        two positive ints.
    """
    if not size:
        return DEFAULT_DIMS
    try:
        w_str, h_str = size.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return DEFAULT_DIMS
    if width <= 0 or height <= 0:
        return DEFAULT_DIMS
    return width, height


def steps_for(model: str, requested: int | None = None) -> int:
    """Resolve the inference step count.

    SDXL-Lightning UNets are distilled for a fixed step count, so the model name
    wins over any per-request override: asking a 4-step UNet for 30 steps
    degrades the image rather than improving it.

    :param model: Lightning variant name.
    :param requested: Per-request override, used only for an unknown variant.
    :returns: Step count to run.
    """
    if model in LIGHTNING_STEPS:
        return LIGHTNING_STEPS[model]
    return requested if requested and requested > 0 else 4


def device_dtype() -> tuple[str, Any]:
    """Resolve the device and dtype: CUDA float16, else MPS, else CPU float32.

    MPS defaults to float16, which is safe because the fp16-fix VAE is swapped
    in; ``MPS_DTYPE=float32`` forces the heavier full-precision path.

    :returns: ``(device, torch dtype)``.
    """
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        want = os.environ.get("MPS_DTYPE", "float16").strip().lower()
        return "mps", (torch.float32 if want == "float32" else torch.float16)
    return "cpu", torch.float32


class SdxlServer:
    """The pipeline and its settings behind the SDXL app.

    One render runs at a time under :attr:`lock`; a diffusers pipeline is not
    safe to call from two threads at once.

    :param model: Lightning variant, a key of :data:`LIGHTNING_STEPS`.
    :param base: SDXL base repo.
    :param offline: Never reach HuggingFace; fail if the weights are not cached.
    :param output_dir: Where ``response_format=filepath`` images go.
    """

    def __init__(
        self,
        *,
        model: str = "sdxl_lightning_4",
        base: str = DEFAULT_BASE,
        offline: bool = False,
        output_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.base = base
        self.offline = offline
        self.output_dir = output_dir or env_output_dir()
        self.pipe: Any = None
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SdxlServer":
        """Build a server from the environment variables in the module docstring.

        :returns: The configured server.
        """
        return cls(
            model=os.environ.get("SDXL_MODEL", "sdxl_lightning_4"),
            base=os.environ.get("SDXL_BASE", DEFAULT_BASE),
            offline=env_flag("SDXL_OFFLINE"),
            output_dir=env_output_dir(),
        )

    def load_pipeline(self) -> Any:
        """Build the SDXL + Lightning-UNet pipeline once and cache it.

        Weights are fetched on first run and cached under ``~/.cache/huggingface``
        unless :attr:`offline` is set.

        :returns: The diffusers pipeline.
        :raises MissingExtraError: When the ``sdxl`` extra is not installed.
        """
        if self.pipe is not None:
            return self.pipe

        torch = _torch()
        diffusers = _import_extra("diffusers", "sdxl")
        hf_hub_download = _import_extra("huggingface_hub", "sdxl").hf_hub_download
        load_file = _import_extra("safetensors.torch", "sdxl").load_file

        device, dtype = device_dtype()
        steps = steps_for(self.model)
        local_only = self.offline
        if not local_only:
            print("[startup] weights are fetched on first run and cached (~7 GB).", flush=True)
        print(f"[startup] loading {self.base} + {self.model} on {device}/{dtype} ...", flush=True)

        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
            self.base, torch_dtype=dtype, local_files_only=local_only
        ).to(device)

        # In float16 the stock SDXL VAE overflows to black images; swap in the fp16-fix VAE.
        if dtype == torch.float16:
            print(f"[startup] loading fp16-fix VAE {VAE_FP16_FIX} ...", flush=True)
            pipe.vae = diffusers.AutoencoderKL.from_pretrained(
                VAE_FP16_FIX, torch_dtype=dtype, local_files_only=local_only
            ).to(device)

        filename = f"sdxl_lightning_{steps}step_unet.safetensors"
        unet_path = hf_hub_download(LIGHTNING_REPO, filename, local_files_only=local_only)
        state = load_file(unet_path, device=device)
        missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[startup] UNet load: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

        # Lightning wants Euler with trailing timestep spacing.
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        try:
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
        except (AttributeError, RuntimeError, NotImplementedError):
            pass

        self.pipe = pipe
        print("[startup] pipeline ready", flush=True)
        return pipe

    def render(
        self,
        prompt: str,
        negative_prompt: str,
        size: str | None,
        seed: int | None,
        steps: int | None,
    ) -> PILImage:
        """Render one image under the lock.

        :param prompt: Text description of the image.
        :param negative_prompt: What to steer away from.
        :param size: ``"WIDTHxHEIGHT"``; one that does not parse gives 1024x1024.
        :param seed: Random seed, or None for a random one.
        :param steps: Used only when the model is not a known Lightning variant.
        :returns: The rendered image.
        """
        with self.lock:
            torch = _torch()
            width, height = parse_size(size)
            pipe = self.load_pipeline()
            device, _ = device_dtype()

            generator = None
            if seed is not None:
                try:
                    generator = torch.Generator(device=device).manual_seed(int(seed))
                except (RuntimeError, TypeError):
                    generator = None

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps_for(self.model, steps),
                guidance_scale=1.0,  # Lightning: no CFG
                generator=generator,
            )
            image = result.images[0]

            # Return the allocator's freed blocks so idle memory drops back
            # between renders.
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            return image


def create_app(server: SdxlServer | None = None) -> Any:
    """Build the FastAPI app.

    On startup the app builds the pipeline (a no-op when ``server.pipe`` is
    already set), so the port opens only once the server can render.
    ``app.state.server`` is *server*.

    :param server: The pipeline and its settings. Default: :meth:`SdxlServer.from_env`.
    :returns: A ``fastapi.FastAPI`` app.
    """
    from fastapi import FastAPI  # noqa: PLC0415 - the `serve` extra
    from pydantic import BaseModel  # noqa: PLC0415 - the `serve` extra

    srv = server or SdxlServer.from_env()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        srv.load_pipeline()
        yield

    app = FastAPI(title="imagewright SDXL-Lightning image server", lifespan=lifespan)
    app.state.server = srv

    class ImageGenRequest(BaseModel):
        model: str = "sdxl_lightning_4"
        prompt: str
        n: int = 1
        size: str = "1024x1024"
        quality: str | None = None
        num_inference_steps: int | None = None
        seed: int | None = None
        response_format: str = "b64_json"
        negative_prompt: str = "blurry, bad quality, distorted"

    @app.get("/health")
    def health():
        """Liveness probe.

        Does no model work, so it answers before the SDXL weights have been
        fetched or a pipeline built.
        """
        return {"status": "ok", "backend": "sdxl-lightning", "model": srv.model}

    @app.get("/v1/models")
    def list_models():
        """OpenAI-compatible single-model listing."""
        return {
            "object": "list",
            "data": [{"id": srv.model, "object": "model", "owned_by": "sdxl-lightning"}],
        }

    @app.post("/v1/images/generations")
    async def generate_image(req: ImageGenRequest):
        """Render in a worker thread and return the image as base64 or a file path."""
        image = await asyncio.to_thread(
            srv.render,
            req.prompt,
            req.negative_prompt,
            req.size,
            req.seed,
            req.num_inference_steps,
        )
        return image_reply(image, req.response_format, srv.output_dir)

    return app
