"""ImageSynthesizer across its three backends (ported from kg_utils, plus unload).

The mflux-serve backend runs against a real HTTP server; the Flux2Klein model
object and the OpenAI client are fakes installed into ``sys.modules``.
"""

from __future__ import annotations

import base64
import sys

import httpx
import pytest
from PIL import Image

from imagewright.client import (
    _DALLE3_SIZES,
    _MFLUX_SIZES,
    ImageSynthesizer,
    MissingExtraError,
    _parse_size,
)
from imagewright.config import ImageBackend, ImageConfig
from tests.conftest import ContractServer, FakeFluxStack, FakeOpenAI


def _serve(url: str, **kw) -> ImageSynthesizer:
    return ImageSynthesizer(ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url=url, **kw))


def _openai(**kw) -> ImageSynthesizer:
    return ImageSynthesizer(
        ImageConfig(
            backend=ImageBackend.OPENAI,
            api_key="test-key",  # pragma: allowlist secret
            **kw,
        )
    )


def _local(**kw) -> ImageSynthesizer:
    return ImageSynthesizer(ImageConfig(backend=ImageBackend.MFLUX_LOCAL, **kw))


# ---------------------------------------------------------------------------
# Size tables and parsing
# ---------------------------------------------------------------------------

_RATIOS = {"1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4"}


def test_mflux_sizes_all_ratios_present() -> None:
    assert set(_MFLUX_SIZES) == _RATIOS
    assert _MFLUX_SIZES["1:1"] == (1024, 1024)


def test_mflux_sizes_orientation() -> None:
    for ratio in ("3:2", "16:9", "4:3"):
        w, h = _MFLUX_SIZES[ratio]
        assert w > h, f"{ratio} should be landscape"
    for ratio in ("2:3", "9:16", "3:4"):
        w, h = _MFLUX_SIZES[ratio]
        assert h > w, f"{ratio} should be portrait"


def test_dalle3_sizes() -> None:
    assert set(_DALLE3_SIZES) == _RATIOS
    assert _DALLE3_SIZES["1:1"] == "1024x1024"
    for ratio in ("3:2", "16:9", "4:3"):
        assert _DALLE3_SIZES[ratio] == "1792x1024"
    for ratio in ("2:3", "9:16", "3:4"):
        assert _DALLE3_SIZES[ratio] == "1024x1792"
    assert set(_DALLE3_SIZES.values()) == {"1024x1024", "1792x1024", "1024x1792"}


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("768x512", (768, 512)),
        ("768X512", (768, 512)),
        (None, None),
        ("", None),
        ("768", None),
        ("axb", None),
        ("0x512", None),
    ],
)
def test_parse_size(size: str | None, expected: tuple[int, int] | None) -> None:
    assert _parse_size(size) == expected


# ---------------------------------------------------------------------------
# mflux-serve backend, against a real server
# ---------------------------------------------------------------------------


def test_serve_posts_the_contract_payload(contract_server: ContractServer) -> None:
    img = _serve(contract_server.url, steps=8).generate("my test prompt", aspect_ratio="1:1")

    assert isinstance(img, Image.Image)
    assert contract_server.requests[-1][:2] == ("POST", "/v1/images/generations")
    assert contract_server.generate_payloads == [
        {
            "prompt": "my test prompt",
            "n": 1,
            "size": "1024x1024",
            "num_inference_steps": 8,
            "response_format": "b64_json",
        }
    ]


def test_serve_strips_trailing_slash(contract_server: ContractServer) -> None:
    _serve(contract_server.url + "/").generate("prompt")
    assert contract_server.requests[-1][1] == "/v1/images/generations"


def test_serve_steps_override_and_seed(contract_server: ContractServer) -> None:
    _serve(contract_server.url, steps=4).generate("prompt", steps=16, seed=42)
    payload = contract_server.generate_payloads[-1]
    assert payload["num_inference_steps"] == 16
    assert payload["seed"] == 42


def test_serve_explicit_size_beats_aspect_ratio(contract_server: ContractServer) -> None:
    _serve(contract_server.url).generate("prompt", aspect_ratio="1:1", size="768x512")
    assert contract_server.generate_payloads[-1]["size"] == "768x512"


def test_serve_raises_on_server_error(contract_server: ContractServer) -> None:
    contract_server.generate_status = 500
    with pytest.raises(httpx.HTTPStatusError):
        _serve(contract_server.url).generate("prompt")


def test_serve_default_server_url_is_localhost_8090() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_SERVE)
    assert cfg.resolved_server_url() == "http://localhost:8090"


def test_generate_b64_returns_png(contract_server: ContractServer) -> None:
    result = _serve(contract_server.url).generate_b64("prompt")
    assert isinstance(result, str)
    assert base64.b64decode(result)[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# OpenAI backend, with a fake client
# ---------------------------------------------------------------------------


def test_openai_default_call(fake_openai: FakeOpenAI) -> None:
    img = _openai().generate("portrait of a Victorian scholar")

    assert isinstance(img, Image.Image)
    assert fake_openai.api_keys == ["test-key"]  # pragma: allowlist secret
    assert fake_openai.calls == [
        {
            "model": "gpt-image-1",
            "prompt": "portrait of a Victorian scholar",
            "n": 1,
            "size": "1536x1024",
        }
    ]


def test_openai_model_override_uses_dalle_sizes(fake_openai: FakeOpenAI) -> None:
    _openai().generate("prompt", model="dall-e-3", aspect_ratio="2:3")
    assert fake_openai.calls[-1]["model"] == "dall-e-3"
    assert fake_openai.calls[-1]["size"] == "1024x1792"


def test_openai_portrait_size(fake_openai: FakeOpenAI) -> None:
    _openai().generate("prompt", aspect_ratio="2:3")
    assert fake_openai.calls[-1]["size"] == "1024x1536"


def test_openai_key_falls_back_to_env(
    fake_openai: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")  # pragma: allowlist secret
    ImageSynthesizer(ImageConfig(backend=ImageBackend.OPENAI)).generate("prompt")
    assert fake_openai.api_keys == ["sk-from-env"]  # pragma: allowlist secret


def test_openai_url_response_is_fetched(
    fake_openai: FakeOpenAI, contract_server: ContractServer
) -> None:
    # A dall-e-3 style reply carries a URL to fetch instead of b64_json.
    fake_openai.b64_json = None
    fake_openai.url = contract_server.url + "/image.png"
    img = _openai().generate("prompt", model="dall-e-3")
    assert isinstance(img, Image.Image)
    assert contract_server.requests[-1][:2] == ("GET", "/image.png")


def test_openai_empty_response_raises(fake_openai: FakeOpenAI) -> None:
    fake_openai.b64_json = None
    with pytest.raises(RuntimeError, match="neither b64_json nor url"):
        _openai().generate("prompt")


# ---------------------------------------------------------------------------
# mflux-local backend, with a fake Flux2Klein
# ---------------------------------------------------------------------------


def test_local_generate_passes_arguments(fake_flux: FakeFluxStack) -> None:
    img = _local(model="my-flux-model", steps=6).generate(
        "a historic seaport at dusk", seed=99, aspect_ratio="16:9"
    )

    assert isinstance(img, Image.Image)
    assert img.size == _MFLUX_SIZES["16:9"]
    assert fake_flux.loads == ["my-flux-model"]
    assert fake_flux.calls == [
        {
            "seed": 99,
            "prompt": "a historic seaport at dusk",
            "width": 1536,
            "height": 864,
            "guidance": 1.0,
            "num_inference_steps": 6,
            "scheduler": "flow_match_euler_discrete",
        }
    ]


def test_local_random_seed_when_none(fake_flux: FakeFluxStack) -> None:
    _local().generate("prompt", seed=None)
    assert isinstance(fake_flux.calls[-1]["seed"], int)


def test_local_default_model(fake_flux: FakeFluxStack) -> None:
    _local().generate("prompt")
    assert fake_flux.loads == ["mlx-community/flux2-klein-4b-4bit"]


def test_local_model_cache_reuses_instance(fake_flux: FakeFluxStack) -> None:
    synth = _local(model="m")
    synth.generate("first call")
    synth.generate("second call")
    assert fake_flux.loads == ["m"]
    assert len(fake_flux.calls) == 2


def test_local_model_change_reloads(fake_flux: FakeFluxStack) -> None:
    synth = _local(model="a")
    synth.generate("prompt")
    synth.generate("prompt", model="b")
    assert fake_flux.loads == ["a", "b"]


# ---------------------------------------------------------------------------
# model_loaded / unload_model
# ---------------------------------------------------------------------------


def test_unload_cycle(fake_flux: FakeFluxStack) -> None:
    synth = _local(model="m")
    assert synth.model_loaded is False
    assert synth.unload_model() is False
    assert fake_flux.cache_clears == 0

    synth.generate("prompt")
    assert synth.model_loaded is True

    assert synth.unload_model() is True
    assert synth.model_loaded is False
    assert fake_flux.cache_clears == 1

    synth.generate("prompt")  # loads again
    assert synth.model_loaded is True
    assert fake_flux.loads == ["m", "m"]


def test_unload_without_mlx(fake_flux: FakeFluxStack, monkeypatch: pytest.MonkeyPatch) -> None:
    synth = _local()
    synth.generate("prompt")
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert synth.unload_model() is True
    assert synth.model_loaded is False
    assert fake_flux.cache_clears == 0


def test_serve_backend_never_loads_a_model(contract_server: ContractServer) -> None:
    synth = _serve(contract_server.url)
    synth.generate("prompt")
    assert synth.model_loaded is False


# ---------------------------------------------------------------------------
# Missing extras
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_extras")
def test_local_without_flux_names_the_extra() -> None:
    with pytest.raises(MissingExtraError, match=r'pip install "imagewright\[flux\]"'):
        _local().generate("prompt")


@pytest.mark.usefixtures("no_extras")
def test_openai_without_openai_names_the_extra() -> None:
    with pytest.raises(MissingExtraError, match=r'pip install "imagewright\[openai\]"'):
        _openai().generate("prompt")


def test_missing_extra_is_an_import_error() -> None:
    assert issubclass(MissingExtraError, ImportError)
