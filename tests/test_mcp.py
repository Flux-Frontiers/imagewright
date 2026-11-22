"""The MCP side: image compression, and the real FastMCP server over an in-memory session."""

import asyncio
import base64
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import Image as MCPImage
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

from imagewright import ImageBackend, ImageConfig, MissingExtraError
from imagewright.mcp_image import MCP_IMAGE_MAX_BYTES, as_mcp_image, compress_image
from imagewright.mcp_server import build_server, main
from tests.conftest import ContractServer, FakeFluxStack


@pytest.fixture
def no_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make mcp unimportable, though this module has already imported it."""
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
        monkeypatch.setitem(sys.modules, name, None)


def noise(width: int, height: int) -> Image.Image:
    """An image that compresses badly, so the size limit actually bites."""
    return Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))


# ---------------------------------------------------------------------------
# compress_image and as_mcp_image
# ---------------------------------------------------------------------------


def test_a_small_image_stays_png():
    data, fmt = compress_image(Image.new("RGB", (64, 48), "white"))
    assert fmt == "png"
    assert Image.open(BytesIO(data)).size == (64, 48)


def test_a_large_image_comes_back_under_the_default_limit():
    image = noise(2048, 1536)  # about 9.4 MB as PNG
    data, fmt = compress_image(image)
    assert fmt == "jpeg"
    assert len(data) <= MCP_IMAGE_MAX_BYTES


def test_jpeg_keeps_full_size_when_it_fits():
    image = noise(512, 512)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85, optimize=True)
    data, fmt = compress_image(image, max_bytes=buf.tell())
    assert fmt == "jpeg"
    assert Image.open(BytesIO(data)).size == (512, 512)


def test_downscales_when_jpeg_alone_is_not_enough():
    data, fmt = compress_image(noise(1024, 768), max_bytes=40_000)
    assert fmt == "jpeg"
    assert len(data) <= 40_000
    width, height = Image.open(BytesIO(data)).size
    assert width < 1024 and abs(width / height - 4 / 3) < 0.05


def test_an_impossible_limit_is_an_error():
    with pytest.raises(ValueError, match="below 10 bytes"):
        compress_image(noise(64, 64), max_bytes=10)


def test_as_mcp_image_returns_image_content():
    result = as_mcp_image(noise(1024, 1024), max_bytes=100_000)
    assert isinstance(result, MCPImage)
    content = result.to_image_content()
    assert content.mimeType == "image/jpeg"
    assert len(base64.b64decode(content.data)) <= 100_000


@pytest.mark.usefixtures("no_mcp")
def test_compression_needs_no_mcp_but_wrapping_does():
    assert compress_image(Image.new("RGB", (8, 8)))[1] == "png"
    with pytest.raises(MissingExtraError, match=r'pip install "imagewright\[mcp\]"'):
        as_mcp_image(Image.new("RGB", (8, 8)))


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


def run(server: Any, call: str, arguments: dict[str, Any] | None = None) -> Any:
    """List the tools, or call one, through a real client session."""

    async def go() -> Any:
        async with create_connected_server_and_client_session(server) as session:
            if call == "list":
                return (await session.list_tools()).tools
            return await session.call_tool(call, arguments or {})

    return asyncio.run(go())


def image_of(result: CallToolResult) -> Image.Image:
    assert not result.isError, result.content
    (content,) = result.content
    assert isinstance(content, ImageContent)
    return Image.open(BytesIO(base64.b64decode(content.data)))


def error_of(result: CallToolResult) -> str:
    assert result.isError
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


def test_the_generate_tool_is_listed():
    (tool,) = run(build_server(), "list")
    assert tool.name == "generate"
    assert set(tool.inputSchema["properties"]) == {"prompt", "size", "seed", "steps"}
    assert tool.inputSchema["required"] == ["prompt"]


def test_generate_returns_an_image_from_the_backend(contract_server: ContractServer):
    config = ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url=contract_server.url)
    result = run(
        build_server(config),
        "generate",
        {"prompt": "a lighthouse", "size": "640x480", "seed": 7, "steps": 3},
    )
    assert image_of(result).size == (4, 4)  # what the contract server returns
    payload = contract_server.generate_payloads[-1]
    assert (payload["prompt"], payload["size"], payload["seed"]) == ("a lighthouse", "640x480", 7)
    assert payload["num_inference_steps"] == 3


def test_generate_finds_the_backend_from_the_environment(
    contract_server: ContractServer, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("IMAGE_ENDPOINT", contract_server.url)
    monkeypatch.setenv("IMAGE_STEPS", "6")
    image_of(run(build_server(), "generate", {"prompt": "p"}))
    payload = contract_server.generate_payloads[-1]
    assert (payload["size"], payload["num_inference_steps"]) == ("1536x1024", 6)


def test_an_in_process_model_loads_once(fake_flux: FakeFluxStack, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IMAGE_BACKEND", "mflux-local")
    server = build_server()

    async def twice() -> list[CallToolResult]:
        async with create_connected_server_and_client_session(server) as session:
            args = {"prompt": "p", "size": "96x64"}
            return [await session.call_tool("generate", args) for _ in range(2)]

    results = asyncio.run(twice())
    assert [image_of(r).size for r in results] == [(96, 64), (96, 64)]
    assert len(fake_flux.loads) == 1


@pytest.mark.usefixtures("no_extras")
def test_no_backend_is_a_tool_error_listing_the_fixes():
    text = error_of(run(build_server(), "generate", {"prompt": "p"}))
    assert "No image backend is available" in text
    assert "imagewright serve" in text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"prompt": "  "}, "prompt is empty"),
        ({"prompt": "x" * 4001}, "longer than 4000"),
        ({"prompt": "p", "size": "huge"}, "not WIDTHxHEIGHT"),
        ({"prompt": "p", "steps": 0}, "steps must be 1-50"),
        ({"prompt": "p", "steps": 51}, "steps must be 1-50"),
    ],
)
def test_bad_arguments_are_rejected_before_rendering(
    contract_server: ContractServer, arguments: dict, message: str
):
    config = ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url=contract_server.url)
    assert message in error_of(run(build_server(config), "generate", arguments))
    assert contract_server.generate_payloads == []


# ---------------------------------------------------------------------------
# The console script
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_mcp")
def test_main_without_mcp_names_the_pip_command(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert 'pip install "imagewright[mcp]"' in capsys.readouterr().err


def test_the_installed_script_serves_over_stdio(tmp_path: Path):
    script = Path(sys.executable).parent / "imagewright-mcp"

    async def go() -> list[str]:
        params = StdioServerParameters(command=str(script), args=[], cwd=str(tmp_path))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return [tool.name for tool in (await session.list_tools()).tools]

    assert asyncio.run(go()) == ["generate"]
