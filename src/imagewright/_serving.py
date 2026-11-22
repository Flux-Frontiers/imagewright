"""Pieces both image servers share: environment parsing, extras checks and the reply body."""

from __future__ import annotations

import base64
import importlib.util
import os
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL.Image import Image as PILImage

from imagewright.client import MissingExtraError

#: Modules the ``serve`` extra installs.
SERVE_MODULES: tuple[str, ...] = ("fastapi", "uvicorn", "pydantic")


def env_flag(name: str) -> bool:
    """Read a yes/no environment variable.

    :param name: Variable name.
    :returns: True when it is ``1``, ``true`` or ``yes`` (any case, spaces ignored).
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def env_output_dir() -> Path:
    """Directory for ``response_format=filepath`` images.

    :returns: ``IMAGE_OUTPUT_DIR``, else ``imagewright_images`` in the system temp directory.
    """
    default = Path(tempfile.gettempdir()) / "imagewright_images"
    return Path(os.environ.get("IMAGE_OUTPUT_DIR") or default)


def require_modules(modules: tuple[str, ...], extras: str, what: str) -> None:
    """Fail with the pip command to run when any of *modules* is not installed.

    :param modules: Top-level module names to look for.
    :param extras: The extras that install them, comma-separated, e.g. ``"serve,flux"``.
    :param what: What needs them, for the message, e.g. ``"the FLUX server"``.
    :raises MissingExtraError: Naming the missing modules and the pip command.
    """
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise MissingExtraError(
            f"{', '.join(missing)} not installed; {what} needs the {extras!r} "
            f'extras: pip install "imagewright[{extras}]"'
        )


def image_reply(image: PILImage, response_format: str, output_dir: Path) -> dict[str, Any]:
    """Build the ``/v1/images/generations`` response body.

    :param image: The rendered image.
    :param response_format: ``"filepath"`` writes a PNG under *output_dir* and returns its
        path; anything else returns the PNG as base64 in ``b64_json``.
    :param output_dir: Where ``filepath`` images go; created when missing.
    :returns: ``{"created": <unix time>, "data": [{"b64_json" | "filepath": ...}]}``.
    """
    if response_format == "filepath":
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{uuid.uuid4().hex}.png"
        image.save(str(out))
        return {"created": int(time.time()), "data": [{"filepath": str(out)}]}

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"created": int(time.time()), "data": [{"b64_json": b64}]}
