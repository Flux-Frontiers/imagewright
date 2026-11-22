"""Find an image server, and choose the backend a caller should use."""

from __future__ import annotations

import importlib.util
import os
import warnings
from dataclasses import replace

import httpx

from imagewright.config import ImageBackend, ImageConfig, env_image_endpoint, image_config_from_env

#: Endpoints probed when nothing is configured, in preference order: FLUX/mflux
#: on 8090, SDXL-Lightning on 8091. These are the ports gutenberg_kg's
#: ``make up`` binds.
DEFAULT_IMAGE_ENDPOINTS: tuple[str, ...] = (
    "http://localhost:8090",
    "http://localhost:8091",
)

_FIXES = """\
No image backend is available. Any one of these fixes it:
  - start an image server on localhost:8090 or localhost:8091
    (it is found automatically);
  - point at a running server: set IMAGE_ENDPOINT=http://HOST:PORT
    or pass --endpoint;
  - render in process on Apple Silicon: pip install "imagewright[flux]";
  - use OpenAI: pip install "imagewright[openai]", set OPENAI_API_KEY
    and IMAGE_BACKEND=openai (or pass --backend openai)."""


class NoImageBackendError(RuntimeError):
    """No backend was named, no server answered, and in-process FLUX is not installed.

    The message lists the ways to fix it.
    """


def discover_image_endpoint(
    candidates: tuple[str, ...] | None = None,
    timeout: float = 1.5,
) -> str | None:
    """Find an image server that is actually up.

    Tries ``/health`` first and falls back to ``/v1/models``: ``/health`` is
    the newer route, and an older server or a third-party mflux-serve only
    answers the model listing. A connection refused, a timeout, or any non-200
    all mean "not this one".

    :param candidates: Base URLs to try, in order. ``None`` tries
        ``IMAGE_ENDPOINT``, then ``GUTENKG_IMAGE_ENDPOINT`` (each when set),
        then :data:`DEFAULT_IMAGE_ENDPOINTS`.
    :param timeout: Per-request timeout in seconds.
    :returns: The first responding base URL, or None if none answered.
    """
    if candidates is None:
        configured = (os.environ.get("IMAGE_ENDPOINT"), os.environ.get("GUTENKG_IMAGE_ENDPOINT"))
        candidates = tuple(url for url in configured if url) + DEFAULT_IMAGE_ENDPOINTS

    for base in candidates:
        root = base.rstrip("/")
        for route in ("/health", "/v1/models"):
            try:
                reply = httpx.get(f"{root}{route}", timeout=timeout)
            except Exception:  # noqa: BLE001 - any failure just means "try the next"
                break  # port is down; no point trying its other route
            if reply.status_code == 200:
                return base
    return None


def resolve_image_config(
    backend: ImageBackend | str | None = None,
    endpoint: str | None = None,
    *,
    timeout: float = 1.5,
) -> ImageConfig:
    """Choose the image backend, in a fixed order, and return its configuration.

    1. *backend* and *endpoint* as given. An endpoint with no backend means
       ``mflux-serve``, whatever ``IMAGE_BACKEND`` says.
    2. ``IMAGE_BACKEND`` and ``IMAGE_ENDPOINT`` (``GUTENKG_IMAGE_ENDPOINT`` as a
       fallback) from the environment, the same way. A configured endpoint is
       used without probing it.
    3. A server that answers on :data:`DEFAULT_IMAGE_ENDPOINTS`.
    4. In-process FLUX (``mflux-local``) if mflux is installed, after a warning
       that no server was found and the model is loading in process.
    5. Otherwise :class:`NoImageBackendError`, whose message lists the fixes.

    Rungs 1-3 never load a model unless ``mflux-local`` is named. A named
    ``mflux-serve`` with no endpoint uses a discovered server, else the default
    ``http://localhost:8090``; it never falls through to in-process FLUX.
    Model, steps and API key always come from :func:`image_config_from_env`.

    :param backend: Backend to use, by enum or by value such as ``"openai"``.
    :param endpoint: Image server base URL for ``mflux-serve``.
    :param timeout: Per-request timeout for discovery probes, in seconds.
    :returns: The configuration to pass to ``ImageSynthesizer``.
    :raises ValueError: When *backend* or ``IMAGE_BACKEND`` is not a known backend,
        or ``IMAGE_STEPS`` is not an integer.
    :raises NoImageBackendError: When nothing is named, found or installed.
    """
    base = image_config_from_env()
    if backend is None and endpoint:
        return replace(base, backend=ImageBackend.MFLUX_SERVE, server_url=endpoint)

    chosen_backend = backend or os.environ.get("IMAGE_BACKEND") or None
    chosen_endpoint = endpoint or env_image_endpoint()

    if chosen_backend is not None:
        try:
            kind = ImageBackend(chosen_backend)
        except ValueError:
            valid = ", ".join(b.value for b in ImageBackend)
            raise ValueError(
                f"unknown image backend {chosen_backend!r}; choose one of: {valid}"
            ) from None
        url = chosen_endpoint
        if kind is ImageBackend.MFLUX_SERVE and not url:
            url = discover_image_endpoint(timeout=timeout) or ""
        return replace(base, backend=kind, server_url=url)

    if chosen_endpoint:
        return replace(base, backend=ImageBackend.MFLUX_SERVE, server_url=chosen_endpoint)

    found = discover_image_endpoint(timeout=timeout)
    if found:
        return replace(base, backend=ImageBackend.MFLUX_SERVE, server_url=found)

    if importlib.util.find_spec("mflux") is not None:
        warnings.warn(
            "No image server found; loading FLUX in process. The first run downloads the weights.",
            UserWarning,
            stacklevel=2,
        )
        return replace(base, backend=ImageBackend.MFLUX_LOCAL)

    raise NoImageBackendError(_FIXES)
