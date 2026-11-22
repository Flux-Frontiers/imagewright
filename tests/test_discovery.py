"""discover_image_endpoint against real servers, and each rung of resolve_image_config."""

from __future__ import annotations

import warnings

import pytest

import imagewright
import imagewright.discovery
from imagewright.config import ImageBackend
from imagewright.discovery import (
    NoImageBackendError,
    discover_image_endpoint,
    resolve_image_config,
)
from tests.conftest import ContractServer, FakeFluxStack, closed_port_url

# ---------------------------------------------------------------------------
# discover_image_endpoint
# ---------------------------------------------------------------------------


def test_discover_finds_a_server_on_health(contract_server: ContractServer) -> None:
    assert discover_image_endpoint((contract_server.url,)) == contract_server.url
    assert [path for _, path, _ in contract_server.requests] == ["/health"]


def test_discover_falls_back_to_models(contract_server: ContractServer) -> None:
    contract_server.health_status = 404  # an older server without /health
    assert discover_image_endpoint((contract_server.url,)) == contract_server.url
    assert [path for _, path, _ in contract_server.requests] == ["/health", "/v1/models"]


def test_discover_skips_an_unhealthy_server(contract_server: ContractServer) -> None:
    contract_server.health_status = 503
    contract_server.models_status = 503
    assert discover_image_endpoint((contract_server.url,)) is None


def test_discover_skips_a_closed_port_and_takes_the_next(
    contract_server: ContractServer,
) -> None:
    candidates = (closed_port_url(), contract_server.url)
    assert discover_image_endpoint(candidates) == contract_server.url


def test_discover_returns_none_when_nothing_answers() -> None:
    assert discover_image_endpoint((closed_port_url(), closed_port_url())) is None


def test_discover_default_probes_env_before_localhost(
    contract_server: ContractServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = closed_port_url()
    monkeypatch.setenv("IMAGE_ENDPOINT", other)
    monkeypatch.setenv("GUTENKG_IMAGE_ENDPOINT", contract_server.url)
    assert discover_image_endpoint() == contract_server.url


def test_discover_default_includes_localhost_ports(
    contract_server: ContractServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        imagewright.discovery, "DEFAULT_IMAGE_ENDPOINTS", (closed_port_url(), contract_server.url)
    )
    assert discover_image_endpoint() == contract_server.url


def test_default_endpoints_are_8090_then_8091() -> None:
    # The autouse fixture patches the module attribute; the package export is the original.
    assert imagewright.DEFAULT_IMAGE_ENDPOINTS == ("http://localhost:8090", "http://localhost:8091")


# ---------------------------------------------------------------------------
# resolve_image_config: one test per rung
# ---------------------------------------------------------------------------


def test_rung_a_explicit_backend(fake_flux: FakeFluxStack) -> None:
    cfg = resolve_image_config("openai")
    assert cfg.backend is ImageBackend.OPENAI


def test_rung_a_explicit_endpoint_means_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_ENDPOINT", "http://env-box:8090")
    cfg = resolve_image_config(endpoint="http://arg-box:8090")
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.server_url == "http://arg-box:8090"


def test_rung_a_explicit_backend_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", "openai")
    assert resolve_image_config(ImageBackend.MFLUX_LOCAL).backend is ImageBackend.MFLUX_LOCAL


def test_rung_a_explicit_serve_without_endpoint_discovers(
    contract_server: ContractServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(imagewright.discovery, "DEFAULT_IMAGE_ENDPOINTS", (contract_server.url,))
    cfg = resolve_image_config("mflux-serve")
    assert cfg.server_url == contract_server.url


def test_rung_a_explicit_serve_never_falls_through_to_flux(fake_flux: FakeFluxStack) -> None:
    cfg = resolve_image_config("mflux-serve")
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.resolved_server_url() == "http://localhost:8090"


def test_rung_a_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="mflux-local, mflux-serve, openai"):
        resolve_image_config("diffusers")


def test_rung_b_env_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", "openai")
    monkeypatch.setenv("IMAGE_MODEL", "dall-e-3")
    cfg = resolve_image_config()
    assert cfg.backend is ImageBackend.OPENAI
    assert cfg.model == "dall-e-3"


def test_rung_b_env_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", "diffusers")
    with pytest.raises(ValueError, match="unknown image backend 'diffusers'"):
        resolve_image_config()


@pytest.mark.parametrize("var", ["IMAGE_ENDPOINT", "GUTENKG_IMAGE_ENDPOINT"])
def test_rung_b_env_endpoint_is_used_without_probing(
    var: str, fake_flux: FakeFluxStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = closed_port_url()  # down, but named: it is used, and nothing loads in process
    monkeypatch.setenv(var, url)
    cfg = resolve_image_config()
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.server_url == url


def test_rung_c_discovered_server_beats_flux(
    contract_server: ContractServer, fake_flux: FakeFluxStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        imagewright.discovery, "DEFAULT_IMAGE_ENDPOINTS", (closed_port_url(), contract_server.url)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = resolve_image_config()
    assert caught == []
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.server_url == contract_server.url


def test_rung_d_flux_in_process_warns(fake_flux: FakeFluxStack) -> None:
    with pytest.warns(UserWarning, match="No image server found; loading FLUX in process"):
        cfg = resolve_image_config()
    assert cfg.backend is ImageBackend.MFLUX_LOCAL
    assert fake_flux.loads == []  # choosing the backend does not load the model


@pytest.mark.usefixtures("no_extras")
def test_rung_e_nothing_available_lists_the_fixes() -> None:
    with pytest.raises(NoImageBackendError) as info:
        resolve_image_config()
    message = str(info.value)
    assert "localhost:8090" in message
    assert "IMAGE_ENDPOINT" in message
    assert 'pip install "imagewright[flux]"' in message
    assert 'pip install "imagewright[openai]"' in message
    assert "OPENAI_API_KEY" in message
