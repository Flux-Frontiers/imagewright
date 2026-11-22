"""``imagewright-mcp``: an MCP server with one ``generate`` tool.

The tool picks its backend with :func:`~imagewright.discovery.resolve_image_config`
on its first call, the same order ``imagewright generate`` uses, and keeps that
:class:`~imagewright.client.ImageSynthesizer` for the life of the server, so an
in-process model loads once. It returns the image through
:func:`~imagewright.mcp_image.as_mcp_image`, under the client's size limit.

Needs the ``mcp`` extra; nothing at module scope imports it, so the console
script loads on a core install and names the pip command instead of failing
with an ImportError.
"""

import asyncio
import sys
import threading
from typing import Any

from PIL.Image import Image as PILImage

from imagewright.client import ImageSynthesizer, MissingExtraError, _import_extra, _parse_size
from imagewright.config import ImageConfig
from imagewright.discovery import resolve_image_config
from imagewright.mcp_image import as_mcp_image

INSTRUCTIONS = (
    "imagewright generates images from text prompts. Call `generate` with a "
    "description of the image; it returns the image. The backend is a local "
    "image server when one is running, else FLUX in process if installed, else "
    "whatever IMAGE_BACKEND names."
)

#: Bounds on the tool's arguments: an MCP client is an external caller.
MAX_PROMPT_CHARS = 4000
MAX_STEPS = 50


def build_server(config: ImageConfig | None = None) -> Any:
    """Build the FastMCP server.

    :param config: Backend to use. Default: chosen by :func:`resolve_image_config`
        on the first ``generate`` call, and kept once it succeeds.
    :returns: A ``mcp.server.fastmcp.FastMCP`` server.
    :raises MissingExtraError: When the ``mcp`` extra is not installed.
    """
    fastmcp = _import_extra("mcp.server.fastmcp", "mcp")
    Image = fastmcp.Image  # the tool's return annotation
    server = fastmcp.FastMCP("imagewright", instructions=INSTRUCTIONS)
    lock = threading.Lock()
    synth: ImageSynthesizer | None = None

    def render(prompt: str, size: str, seed: int | None, steps: int | None) -> PILImage:
        nonlocal synth
        with lock:  # one render at a time; an in-process model is not thread-safe
            if synth is None:
                synth = ImageSynthesizer(config or resolve_image_config())
            return synth.generate(prompt, size=size, seed=seed, steps=steps)

    @server.tool()
    async def generate(
        prompt: str,
        size: str = "1536x1024",
        seed: int | None = None,
        steps: int | None = None,
    ) -> Image:
        """Generate an image from a text prompt and return it.

        :param prompt: Description of the image, e.g. "a lighthouse at dusk, oil painting".
        :param size: Output size "WIDTHxHEIGHT" (default landscape 1536x1024). FLUX
            servers render any size; the OpenAI backend picks its nearest size.
        :param seed: Integer seed for a reproducible image; random when omitted.
        :param steps: Inference steps, 1-50: 4 is fast, 25 is higher quality.
            Default: IMAGE_STEPS, else 4.
        """
        if not prompt.strip():
            raise ValueError("prompt is empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
        if _parse_size(size) is None:
            raise ValueError(f"size {size!r} is not WIDTHxHEIGHT, e.g. 1536x1024")
        if steps is not None and not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be 1-{MAX_STEPS}, not {steps}")
        image = await asyncio.to_thread(render, prompt, size, seed, steps)
        return as_mcp_image(image)

    return server


def main() -> None:
    """Run ``imagewright-mcp`` over stdio."""
    try:
        server = build_server()
    except MissingExtraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    server.run()
