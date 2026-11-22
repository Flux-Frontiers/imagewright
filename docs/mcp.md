# MCP server

`imagewright-mcp` is an MCP server over stdio with one tool. It needs the
`mcp` extra:

```bash
pip install "imagewright[mcp]"
```

Without it, the command exits with status 1:

```text
Error: mcp is not installed; this backend needs the 'mcp' extra: pip install "imagewright[mcp]"
```

## Add it to Claude Code

Put this in the project's `.mcp.json`:

```json
{
  "mcpServers": {
    "imagewright": { "command": "imagewright-mcp" }
  }
}
```

To send it to a particular server, set `IMAGE_ENDPOINT` in the entry's `env`:

```json
{
  "mcpServers": {
    "imagewright": {
      "command": "imagewright-mcp",
      "env": { "IMAGE_ENDPOINT": "http://127.0.0.1:8090" }
    }
  }
}
```

## The tool

### `generate(prompt, size, seed, steps)`

Generates an image from a text prompt and returns it.

| Argument | Type | Default | Bounds |
|---|---|---|---|
| `prompt` | string | required | Not empty; at most 4000 characters |
| `size` | string | `"1536x1024"` | `WIDTHxHEIGHT`, two positive integers |
| `seed` | integer | random | none |
| `steps` | integer | `IMAGE_STEPS`, else 4 | 1-50 |

An argument out of bounds fails the call with a message that names the
bound, for example `steps must be 1-50, not 80`. The OpenAI backend ignores
`size` and `steps`.

The tool chooses its backend on its first call, in the order on
[Choose a backend](backends.md), and keeps it for the life of the server, so
an in-process model loads once. A call that fails before a backend is chosen
does not fix the choice; the next call tries again. Renders run one at a
time.

The tool returns the image as PNG, or as JPEG when the PNG is over
`MCP_IMAGE_MAX_BYTES` (3,750,000 bytes, which is 5 MB once base64-encoded,
the usual MCP client limit).

## Return an image from your own MCP tool

`as_mcp_image()` lets another package's FastMCP tool return an image the same
way:

```python
from mcp.server.fastmcp import FastMCP, Image

from imagewright import as_mcp_image

server = FastMCP("example")


@server.tool()
def render() -> Image:
    pil_image = ...  # any PIL image
    return as_mcp_image(pil_image)
```

`as_mcp_image(image, max_bytes=MCP_IMAGE_MAX_BYTES)` returns a
`mcp.server.fastmcp.Image` holding the encoded bytes. It needs the `mcp`
extra.

`compress_image(image, max_bytes=MCP_IMAGE_MAX_BYTES)` does the encoding and
needs only Pillow. It returns `(data, format)`, where format is `"png"` or
`"jpeg"`, and tries these in order:

1. PNG.
2. JPEG at quality 85, then 70, then 50.
3. JPEG at quality 70, shrinking the image to three quarters of its width and
   height each round until it fits.

It raises `ValueError` when even the smallest image does not fit.
