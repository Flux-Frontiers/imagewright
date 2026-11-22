"""ImageBackend, ImageConfig and image_config_from_env (ported from kg_utils)."""

from __future__ import annotations

import pytest

from imagewright.config import _IMAGE_DEFAULTS, ImageBackend, ImageConfig, image_config_from_env


def test_image_backend_values() -> None:
    assert ImageBackend.MFLUX_LOCAL.value == "mflux-local"
    assert ImageBackend.MFLUX_SERVE.value == "mflux-serve"
    assert ImageBackend.OPENAI.value == "openai"


def test_image_backend_from_string() -> None:
    assert ImageBackend("mflux-local") is ImageBackend.MFLUX_LOCAL
    assert ImageBackend("mflux-serve") is ImageBackend.MFLUX_SERVE
    assert ImageBackend("openai") is ImageBackend.OPENAI


def test_image_config_mflux_serve_defaults() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_SERVE)
    assert cfg.resolved_server_url() == "http://localhost:8090"
    assert cfg.resolved_model() == _IMAGE_DEFAULTS[ImageBackend.MFLUX_SERVE]["model"]


def test_image_config_mflux_local_defaults() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_LOCAL)
    assert cfg.resolved_model() == "mlx-community/flux2-klein-4b-4bit"
    assert cfg.resolved_server_url() == ""


def test_image_config_openai_defaults() -> None:
    cfg = ImageConfig(backend=ImageBackend.OPENAI)
    assert cfg.resolved_model() == "gpt-image-1"
    assert cfg.resolved_server_url() == ""


def test_image_config_server_url_override() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url="http://gpu-box:8090")
    assert cfg.resolved_server_url() == "http://gpu-box:8090"


def test_image_config_model_override() -> None:
    cfg = ImageConfig(backend=ImageBackend.OPENAI, model="dall-e-2")
    assert cfg.resolved_model() == "dall-e-2"


def test_image_config_steps_default() -> None:
    assert ImageConfig().steps == 4


# ---------------------------------------------------------------------------
# image_config_from_env (the autouse `hermetic` fixture clears the env)
# ---------------------------------------------------------------------------


def test_image_config_from_env_defaults() -> None:
    cfg = image_config_from_env()
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.server_url == ""
    assert cfg.api_key == ""
    assert cfg.model == ""
    assert cfg.steps == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("mflux-local", ImageBackend.MFLUX_LOCAL),
        ("openai", ImageBackend.OPENAI),
        ("diffusers", ImageBackend.MFLUX_SERVE),  # unknown -> default
    ],
)
def test_image_config_from_env_backend(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: ImageBackend
) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", value)
    assert image_config_from_env().backend is expected


def test_image_config_from_env_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_ENDPOINT", "http://gpu-box:8090")
    assert image_config_from_env().server_url == "http://gpu-box:8090"


def test_image_config_from_env_legacy_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUTENKG_IMAGE_ENDPOINT", "http://old-box:8090")
    assert image_config_from_env().server_url == "http://old-box:8090"


def test_image_config_from_env_endpoint_beats_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_ENDPOINT", "http://new-box:8090")
    monkeypatch.setenv("GUTENKG_IMAGE_ENDPOINT", "http://old-box:8090")
    assert image_config_from_env().server_url == "http://new-box:8090"


def test_image_config_from_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "dall-e-2")
    assert image_config_from_env().model == "dall-e-2"


def test_image_config_from_env_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_STEPS", "8")
    assert image_config_from_env().steps == 8


def test_image_config_from_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "img-key-abc")
    assert image_config_from_env().api_key == "img-key-abc"  # pragma: allowlist secret


def test_image_config_from_env_openai_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    assert image_config_from_env().api_key == "sk-openai-fallback"  # pragma: allowlist secret


def test_image_config_from_env_image_api_key_beats_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "img-wins")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-loses")
    assert image_config_from_env().api_key == "img-wins"  # pragma: allowlist secret


def test_image_config_from_env_legacy_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "flux2-klein-4b")
    assert image_config_from_env().model == "flux2-klein-4b"


def test_image_config_from_env_image_model_beats_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "dall-e-3")
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "old-flux-model")
    assert image_config_from_env().model == "dall-e-3"
