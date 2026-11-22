"""The command line as an installed user meets it."""

from importlib.metadata import version
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

import imagewright
from imagewright.cli import cli
from tests.conftest import ContractServer, FakeFluxStack, closed_port_url


def test_version_matches_the_installed_distribution():
    assert imagewright.__version__ == version("imagewright")


def test_cli_reports_its_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"imagewright, version {imagewright.__version__}"


def test_cli_help_lists_generate():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Generate images" in result.output
    assert "generate" in result.output


def test_generate_against_a_server(contract_server: ContractServer, tmp_path: Path):
    out = tmp_path / "out.png"
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            "a lighthouse",
            "--endpoint",
            contract_server.url,
            "--size",
            "640x480",
            "--seed",
            "7",
            "--steps",
            "3",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(out)
    assert Image.open(out).format == "PNG"
    payload = contract_server.generate_payloads[-1]
    assert payload["prompt"] == "a lighthouse"
    assert payload["size"] == "640x480"
    assert payload["seed"] == 7
    assert payload["num_inference_steps"] == 3


def test_generate_default_output_name(
    contract_server: ContractServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMAGE_ENDPOINT", contract_server.url)
    result = CliRunner().invoke(cli, ["generate", "x"])
    assert result.exit_code == 0, result.output
    written = Path(result.output.strip())
    assert written.name.startswith("imagewright-")
    assert written.suffix == ".png"
    assert (tmp_path / written).is_file()


def test_generate_in_process_warns_and_writes(
    fake_flux: FakeFluxStack, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["generate", "x", "-o", "flux.png"])
    assert result.exit_code == 0, result.output
    assert "warning: No image server found; loading FLUX in process" in result.stderr
    assert result.stdout.strip() == "flux.png"
    assert fake_flux.loads == ["mlx-community/flux2-klein-4b-4bit"]


@pytest.mark.usefixtures("no_extras")
def test_generate_with_nothing_available_lists_the_fixes():
    result = CliRunner().invoke(cli, ["generate", "x"])
    assert result.exit_code == 1
    assert "No image backend is available" in result.output
    assert 'pip install "imagewright[flux]"' in result.output


@pytest.mark.usefixtures("no_extras")
@pytest.mark.parametrize(("backend", "extra"), [("mflux-local", "flux"), ("openai", "openai")])
def test_generate_missing_extra_names_it(backend: str, extra: str):
    result = CliRunner().invoke(cli, ["generate", "x", "--backend", backend])
    assert result.exit_code == 1
    assert f'pip install "imagewright[{extra}]"' in result.output


def test_generate_server_down_is_a_clean_error():
    url = closed_port_url()
    result = CliRunner().invoke(cli, ["generate", "x", "--endpoint", url])
    assert result.exit_code == 1
    assert f"image server {url} failed" in result.output


def test_generate_rejects_a_bad_size():
    result = CliRunner().invoke(cli, ["generate", "x", "--size", "big"])
    assert result.exit_code == 2
    assert "'big' is not WIDTHxHEIGHT" in result.output
