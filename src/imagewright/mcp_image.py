"""Fit a PIL image under an MCP client's size limit and wrap it as an MCP image.

:func:`compress_image` needs only Pillow. :func:`as_mcp_image` also needs the
``mcp`` extra, imported only when it is called, so other packages (kgrag's
``kg_imagine`` tool, for one) can return an image the same way.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image
from PIL.Image import Image as PILImage

from imagewright.client import _import_extra

#: Largest encoded image returned by default. MCP clients cap an image at 5 MB
#: after base64, which grows the bytes by a third: 3.75 MB raw is 5 MB encoded.
MCP_IMAGE_MAX_BYTES = 3_750_000

_JPEG_QUALITIES = (85, 70, 50)
_DOWNSCALE_QUALITY = 70
_DOWNSCALE_STEP = 0.75


def _jpeg(image: PILImage, quality: int) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def compress_image(image: PILImage, max_bytes: int = MCP_IMAGE_MAX_BYTES) -> tuple[bytes, str]:
    """Encode *image* in at most *max_bytes*, keeping as much of it as the limit allows.

    Tries, in order: PNG; JPEG at quality 85, 70 and 50; then JPEG at quality
    70, shrinking the image to three quarters of its size each round until it
    fits.

    :param image: The image to encode.
    :param max_bytes: Size limit for the encoded bytes.
    :returns: ``(data, format)``, where format is ``"png"`` or ``"jpeg"``.
    :raises ValueError: When even a 1-pixel-wide JPEG exceeds *max_bytes*.
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    if buf.tell() <= max_bytes:
        return buf.getvalue(), "png"

    for quality in _JPEG_QUALITIES:
        data = _jpeg(image, quality)
        if len(data) <= max_bytes:
            return data, "jpeg"

    small = image
    while small.width > 1 and small.height > 1:
        size = (
            max(1, int(small.width * _DOWNSCALE_STEP)),
            max(1, int(small.height * _DOWNSCALE_STEP)),
        )
        small = image.resize(size, Image.Resampling.LANCZOS)
        data = _jpeg(small, _DOWNSCALE_QUALITY)
        if len(data) <= max_bytes:
            return data, "jpeg"
    raise ValueError(f"image could not be compressed below {max_bytes} bytes")


def as_mcp_image(image: PILImage, max_bytes: int = MCP_IMAGE_MAX_BYTES) -> Any:
    """Return *image* as an MCP image a FastMCP tool can return, within *max_bytes*.

    :param image: The image to return.
    :param max_bytes: Size limit for the encoded bytes; see :func:`compress_image`.
    :returns: A ``mcp.server.fastmcp.Image`` holding the encoded bytes.
    :raises MissingExtraError: When the ``mcp`` extra is not installed.
    """
    data, fmt = compress_image(image, max_bytes)
    return _import_extra("mcp.server.fastmcp", "mcp").Image(data=data, format=fmt)
