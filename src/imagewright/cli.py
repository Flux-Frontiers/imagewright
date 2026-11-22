"""The ``imagewright`` command line."""

import click

from imagewright import __version__


@click.group()
@click.version_option(__version__, prog_name="imagewright")
def cli() -> None:
    """Generate images through a local server, an in-process model or OpenAI."""
