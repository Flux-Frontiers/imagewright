"""imagewright -- image generation over one HTTP contract.

A client for ``/v1/images/generations``, local FLUX and SDXL-Lightning
servers that answer it, and an MCP tool. Nothing here depends on a knowledge
graph.
"""

from importlib.metadata import version

from imagewright.client import ImageSynthesizer, MissingExtraError
from imagewright.config import ImageBackend, ImageConfig, image_config_from_env
from imagewright.discovery import (
    DEFAULT_IMAGE_ENDPOINTS,
    NoImageBackendError,
    discover_image_endpoint,
    resolve_image_config,
)
from imagewright.mcp_image import MCP_IMAGE_MAX_BYTES, as_mcp_image, compress_image

__version__ = version("imagewright")

__all__ = [
    "DEFAULT_IMAGE_ENDPOINTS",
    "MCP_IMAGE_MAX_BYTES",
    "ImageBackend",
    "ImageConfig",
    "ImageSynthesizer",
    "MissingExtraError",
    "NoImageBackendError",
    "__version__",
    "as_mcp_image",
    "compress_image",
    "discover_image_endpoint",
    "image_config_from_env",
    "resolve_image_config",
]
