"""The command line as an installed user meets it."""

from importlib.metadata import version

from click.testing import CliRunner

import imagewright
from imagewright.cli import cli


def test_version_matches_the_installed_distribution():
    assert imagewright.__version__ == version("imagewright")


def test_cli_reports_its_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"imagewright, version {imagewright.__version__}"


def test_cli_help_lists_no_commands_yet():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Generate images" in result.output
