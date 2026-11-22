"""The command line as an installed user meets it."""

import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import uvicorn
from click.testing import CliRunner
from PIL import Image

import imagewright
from imagewright.cli import cli
from tests.conftest import ContractServer, FakeFluxStack, FakeSdxlStack, closed_port_url


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


@pytest.mark.parametrize(
    ("var", "value"), [("IMAGE_BACKEND", "diffusers"), ("IMAGE_STEPS", "many")]
)
def test_generate_bad_environment_is_a_clean_error(
    var: str, value: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(var, value)
    result = CliRunner().invoke(cli, ["generate", "x"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.output.startswith("Error: ")


# ---------------------------------------------------------------------------
# imagewright serve
# ---------------------------------------------------------------------------


@pytest.fixture
def uvicorn_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record ``uvicorn.run`` instead of serving forever; the app is still built for real."""
    runs: list[dict] = []
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, host, port: runs.append({"app": app, "host": host, "port": port}),
    )
    return runs


def test_serve_flux_binds_localhost_8090_by_default(fake_flux: FakeFluxStack, uvicorn_runs):
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 0, result.output
    (run,) = uvicorn_runs
    assert (run["host"], run["port"]) == ("127.0.0.1", 8090)
    assert run["app"].title == "imagewright FLUX image server"


def test_serve_flux_reads_the_environment(
    fake_flux: FakeFluxStack, uvicorn_runs, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MFLUX_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("MFLUX_SERVER_PORT", "9000")
    monkeypatch.setenv("IMAGE_IDLE_UNLOAD_S", "300")
    assert CliRunner().invoke(cli, ["serve"]).exit_code == 0
    (run,) = uvicorn_runs
    assert (run["host"], run["port"]) == ("0.0.0.0", 9000)
    assert run["app"].state.server.idle_unload_s == 300


def test_serve_options_beat_the_environment(
    fake_flux: FakeFluxStack, uvicorn_runs, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MFLUX_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("MFLUX_SERVER_PORT", "9000")
    result = CliRunner().invoke(cli, ["serve", "--host", "10.0.0.5", "--port", "8092"])
    assert result.exit_code == 0, result.output
    assert (uvicorn_runs[0]["host"], uvicorn_runs[0]["port"]) == ("10.0.0.5", 8092)


def test_serve_sdxl_binds_localhost_8091_by_default(fake_sdxl: FakeSdxlStack, uvicorn_runs):
    result = CliRunner().invoke(cli, ["serve", "--backend", "sdxl"])
    assert result.exit_code == 0, result.output
    (run,) = uvicorn_runs
    assert (run["host"], run["port"]) == ("127.0.0.1", 8091)
    assert run["app"].title == "imagewright SDXL-Lightning image server"
    assert fake_sdxl.pretrained == []  # the pipeline loads at startup, not when the app is built


def test_serve_sdxl_reads_its_own_variables(
    fake_sdxl: FakeSdxlStack, uvicorn_runs, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MFLUX_SERVER_PORT", "9000")
    monkeypatch.setenv("SDXL_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("SDXL_SERVER_PORT", "9001")
    assert CliRunner().invoke(cli, ["serve", "--backend", "sdxl"]).exit_code == 0
    assert (uvicorn_runs[0]["host"], uvicorn_runs[0]["port"]) == ("0.0.0.0", 9001)


@pytest.mark.usefixtures("no_extras")
def test_serve_flux_without_mflux_names_the_pip_command(uvicorn_runs):
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 1
    assert 'pip install "imagewright[serve,flux]"' in result.output
    assert "Traceback" not in result.output
    assert uvicorn_runs == []


def test_serve_without_the_serve_extra_names_it(
    fake_flux: FakeFluxStack, uvicorn_runs, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 1
    assert "fastapi not installed" in result.output
    assert 'pip install "imagewright[serve,flux]"' in result.output


def test_serve_sdxl_without_torch_names_the_pip_command(monkeypatch: pytest.MonkeyPatch):
    for name in ("torch", "diffusers"):
        monkeypatch.setitem(sys.modules, name, None)
    result = CliRunner().invoke(cli, ["serve", "--backend", "sdxl"])
    assert result.exit_code == 1
    assert 'pip install "imagewright[serve,sdxl]"' in result.output


def test_serve_rejects_an_unknown_backend():
    result = CliRunner().invoke(cli, ["serve", "--backend", "dalle"])
    assert result.exit_code == 2
