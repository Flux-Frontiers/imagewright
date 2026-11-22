"""Image backend configuration: the backend enum, the config dataclass and its env factory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class ImageBackend(str, Enum):  # noqa: UP042 - StrEnum would change str() of a member
    """Image generation backend."""

    MFLUX_LOCAL = "mflux-local"  # in-process Flux2Klein via mflux (Apple Silicon only)
    MFLUX_SERVE = "mflux-serve"  # HTTP client for a server answering /v1/images/generations
    OPENAI = "openai"  # OpenAI gpt-image-1 -- requires OPENAI_API_KEY / IMAGE_API_KEY


# Per-backend defaults, filled in when the user has not provided an override.
_IMAGE_DEFAULTS: dict[ImageBackend, dict[str, str]] = {
    ImageBackend.MFLUX_LOCAL: {
        "model": "mlx-community/flux2-klein-4b-4bit",
    },
    ImageBackend.MFLUX_SERVE: {
        "server_url": "http://localhost:8090",
        "model": "flux2-klein-4b",
    },
    ImageBackend.OPENAI: {
        "model": "gpt-image-1",
    },
}


@dataclass
class ImageConfig:
    """Configuration for an image synthesis backend.

    :param backend: Which image provider to use.
    :param server_url: Image server base URL override (empty = use backend default).
    :param api_key: OpenAI API key (empty = read from OPENAI_API_KEY at call time).
    :param model: Model-id override (empty = use backend default).
    :param steps: Inference steps for mflux backends (ignored for OpenAI).
    """

    backend: ImageBackend = ImageBackend.MFLUX_SERVE
    server_url: str = ""
    api_key: str = ""
    model: str = ""
    steps: int = 4

    def resolved_server_url(self) -> str:
        """Return the server URL, or the backend's default when none is set.

        :returns: Base URL, or an empty string for backends without a server.
        """
        return self.server_url or _IMAGE_DEFAULTS[self.backend].get("server_url", "")

    def resolved_model(self) -> str:
        """Return the model id, or the backend's default when none is set.

        :returns: Model id.
        """
        return self.model or _IMAGE_DEFAULTS[self.backend].get("model", "")


def env_image_endpoint() -> str:
    """Return the image server URL configured in the environment.

    :returns: ``IMAGE_ENDPOINT``, else ``GUTENKG_IMAGE_ENDPOINT``, else an empty string.
    """
    return os.environ.get("IMAGE_ENDPOINT") or os.environ.get("GUTENKG_IMAGE_ENDPOINT") or ""


def image_config_from_env() -> ImageConfig:
    """Build an ImageConfig from environment variables.

    Reads ``IMAGE_BACKEND`` (default ``mflux-serve``; an unknown value also gives
    ``mflux-serve``), ``IMAGE_ENDPOINT``, ``IMAGE_API_KEY``, ``IMAGE_MODEL`` and
    ``IMAGE_STEPS``. ``GUTENKG_IMAGE_ENDPOINT`` and ``GUTENKG_IMAGE_MODEL`` are
    fallbacks for the endpoint and model, and ``OPENAI_API_KEY`` for the key.

    :returns: The configuration the environment describes.
    """
    backend_str = os.environ.get("IMAGE_BACKEND", "mflux-serve")
    try:
        backend = ImageBackend(backend_str)
    except ValueError:
        backend = ImageBackend.MFLUX_SERVE

    api_key = os.environ.get("IMAGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    model = os.environ.get("IMAGE_MODEL") or os.environ.get("GUTENKG_IMAGE_MODEL") or ""
    steps = int(os.environ.get("IMAGE_STEPS", "4"))
    return ImageConfig(
        backend=backend, server_url=env_image_endpoint(), api_key=api_key, model=model, steps=steps
    )
