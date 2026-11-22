"""imagewright -- image generation over one HTTP contract.

A client for ``/v1/images/generations``, local FLUX and SDXL-Lightning
servers that answer it, and an MCP tool. Nothing here depends on a knowledge
graph.
"""

from importlib.metadata import version

__version__ = version("imagewright")

__all__ = ["__version__"]
