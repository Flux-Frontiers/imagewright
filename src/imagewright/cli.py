"""The ``imagewright`` command line."""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import click
import httpx

from imagewright import __version__
from imagewright.client import ImageSynthesizer, MissingExtraError, _parse_size
from imagewright.config import ImageBackend
from imagewright.discovery import NoImageBackendError, resolve_image_config


@click.group()
@click.version_option(__version__, prog_name="imagewright")
def cli() -> None:
    """Generate images through a local server, an in-process model or OpenAI."""


@cli.command()
@click.argument("prompt")
@click.option("--size", metavar="WxH", help="Output size, e.g. 1536x1024 (default 3:2).")
@click.option("--seed", type=int, help="Random seed for a reproducible image.")
@click.option("--steps", type=int, help="Inference steps (FLUX backends; default 4).")
@click.option(
    "--backend",
    type=click.Choice([b.value for b in ImageBackend]),
    help="Backend to use. Default: IMAGE_BACKEND, else a server found locally, "
    "else in-process FLUX if installed.",
)
@click.option("--endpoint", metavar="URL", help="Image server base URL (implies mflux-serve).")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="PNG path to write. Default: imagewright-YYYYMMDD-HHMMSS.png here.",
)
def generate(
    prompt: str,
    size: str | None,
    seed: int | None,
    steps: int | None,
    backend: str | None,
    endpoint: str | None,
    output: Path | None,
) -> None:
    """Generate an image from PROMPT, write it as a PNG and print its path.

    \f
    :param prompt: Text description of the image.
    :param size: ``WIDTHxHEIGHT``, or None for the 3:2 default.
    :param seed: Random seed, or None for a random one.
    :param steps: Inference steps, or None for the configured default.
    :param backend: Backend name, or None to choose one automatically.
    :param endpoint: Image server base URL, or None.
    :param output: Where to write the PNG, or None for a timestamped name.
    """
    if size is not None and _parse_size(size) is None:
        raise click.BadParameter(f"{size!r} is not WIDTHxHEIGHT", param_hint="--size")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            config = resolve_image_config(backend, endpoint)
        except (NoImageBackendError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
    for w in caught:
        click.echo(f"warning: {w.message}", err=True)
    try:
        image = ImageSynthesizer(config).generate(prompt, size=size, seed=seed, steps=steps)
    except MissingExtraError as exc:
        raise click.ClickException(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise click.ClickException(
            f"image server {config.resolved_server_url()} failed: {exc}"
        ) from exc

    path = output or Path(f"imagewright-{datetime.now():%Y%m%d-%H%M%S}.png")
    image.save(path, format="PNG")
    click.echo(str(path))


def _server_module(backend: str):
    """The server module for ``backend``, imported only when a command needs it.

    :param backend: ``flux`` or ``sdxl``.
    """
    if backend == "flux":
        from imagewright import flux_server  # noqa: PLC0415 - only the server commands need it

        return flux_server
    from imagewright import sdxl_server  # noqa: PLC0415 - only the server commands need it

    return sdxl_server


@cli.command()
@click.option(
    "--backend",
    type=click.Choice(["flux", "sdxl"]),
    default="flux",
    show_default=True,
    help="flux: FLUX.2 Klein through mflux (Apple Silicon). sdxl: SDXL-Lightning "
    "through diffusers (CUDA, MPS or CPU).",
)
@click.option(
    "--host",
    help="Address to bind. Default: MFLUX_SERVER_HOST (SDXL_SERVER_HOST for sdxl), "
    "else 127.0.0.1. The server has no authentication; bind 0.0.0.0 only on a "
    "network you trust.",
)
@click.option(
    "--port",
    type=click.IntRange(0, 65535),
    help="Port to bind. Default: MFLUX_SERVER_PORT (SDXL_SERVER_PORT for sdxl), "
    "else 8090 (8091 for sdxl).",
)
def serve(backend: str, host: str | None, port: int | None) -> None:
    """Run an image server that answers /v1/images/generations.

    The FLUX server needs the serve and flux extras; the SDXL server needs serve
    and sdxl.

    \f
    :param backend: ``flux`` or ``sdxl``.
    :param host: Bind address, or None for the environment or 127.0.0.1.
    :param port: Bind port, or None for the environment or the backend's default.
    """
    server = _server_module(backend)
    env_prefix = "MFLUX_SERVER" if backend == "flux" else "SDXL_SERVER"
    try:
        server.require_extras()
    except MissingExtraError as exc:
        raise click.ClickException(str(exc)) from exc

    host = host or os.environ.get(f"{env_prefix}_HOST") or "127.0.0.1"
    if port is None:
        port = int(os.environ.get(f"{env_prefix}_PORT") or server.DEFAULT_PORT)

    import uvicorn  # noqa: PLC0415 - the `serve` extra

    uvicorn.run(server.create_app(), host=host, port=port)


@cli.group()
def service() -> None:
    """Run an image server under launchd, so it starts by itself (macOS).

    `install --boot` makes a LaunchDaemon that starts at boot with nobody
    logged in (needs sudo); without --boot it makes a LaunchAgent that starts
    at login.
    """


_BACKEND = click.option(
    "--backend", type=click.Choice(["flux", "sdxl"]), default="flux", show_default=True
)
_BOOT = click.option(
    "--boot",
    is_flag=True,
    help="A LaunchDaemon that starts at boot, before login. Needs sudo.",
)


def _spec(backend: str, boot: bool, host: str, port: int | None, idle_unload: int):
    """Build the service spec, refusing anything but macOS.

    :param backend: ``flux`` or ``sdxl``.
    :param boot: True for a boot service.
    :param host: Bind address.
    :param port: Bind port, or None for the backend's default.
    :param idle_unload: FLUX idle-unload seconds.
    """
    from imagewright import service as svc  # noqa: PLC0415 - macOS only

    if sys.platform != "darwin":
        raise click.ClickException("`imagewright service` uses launchd and runs on macOS only")
    user, home = svc.target_user(boot)
    return svc.ServiceSpec(
        backend=backend,
        boot=boot,
        user=user,
        home=home,
        executable=svc.imagewright_executable(),
        host=host,
        port=port or svc.DEFAULT_PORTS[backend],
        idle_unload_s=idle_unload,
    )


@service.command("install")
@_BACKEND
@_BOOT
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Address to bind. 0.0.0.0 serves the LAN; the server has no authentication.",
)
@click.option("--port", type=click.IntRange(1, 65535), help="Port. Default 8090 (8091 for sdxl).")
@click.option(
    "--idle-unload",
    type=click.IntRange(0),
    default=0,
    show_default=True,
    metavar="SECONDS",
    help="FLUX unloads its model after this many idle seconds; 0 keeps it loaded.",
)
@click.option("--dry-run", is_flag=True, help="Print the property list and change nothing.")
def service_install(
    backend: str, boot: bool, host: str, port: int | None, idle_unload: int, dry_run: bool
) -> None:
    """Install the server as a launchd service, start it and wait for /health.

    \f
    :param backend: ``flux`` or ``sdxl``.
    :param boot: True for a LaunchDaemon, False for a LaunchAgent.
    :param host: Bind address.
    :param port: Bind port, or None for the backend's default.
    :param idle_unload: FLUX idle-unload seconds.
    :param dry_run: Print the property list only.
    """
    import plistlib  # noqa: PLC0415

    from imagewright import service as svc  # noqa: PLC0415 - macOS only

    spec = _spec(backend, boot, host, port, idle_unload)
    if dry_run:
        click.echo(f"# {spec.plist_path}")
        click.echo(plistlib.dumps(svc.build_plist(spec)).decode())
        return
    server = _server_module(backend)
    try:
        server.require_extras()
    except MissingExtraError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        health = svc.install(spec)
    except svc.ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"installed {spec.plist_path}")
    click.echo(f"log {spec.log_path}")
    click.echo(f"{spec.health_url} {json.dumps(health)}")


@service.command("uninstall")
@_BACKEND
@_BOOT
def service_uninstall(backend: str, boot: bool) -> None:
    """Stop the launchd service and remove its property list.

    \f
    :param backend: ``flux`` or ``sdxl``.
    :param boot: True for the LaunchDaemon, False for the LaunchAgent.
    """
    from imagewright import service as svc  # noqa: PLC0415 - macOS only

    spec = _spec(backend, boot, "127.0.0.1", None, 0)
    try:
        removed = svc.uninstall(spec)
    except svc.ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"removed {spec.plist_path}" if removed else f"no {spec.plist_path}; launchd unloaded"
    )
