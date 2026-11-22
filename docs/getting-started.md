# Install and render an image

imagewright needs Python 3.12 or 3.13. The core install is a client for a
running image server; extras add the backends and the servers.

## Install the package

```bash
pip install imagewright
```

Confirm the install:

```bash
imagewright --version
```

```text
imagewright, version 0.1.0
```

## Add the extras you need

Each extra is a user-facing feature. Install the ones you use:

```bash
pip install "imagewright[flux]"        # FLUX.2 Klein in process (Apple Silicon)
pip install "imagewright[openai]"      # OpenAI gpt-image-1
pip install "imagewright[serve,flux]"  # the FLUX server
pip install "imagewright[serve,sdxl]"  # the SDXL-Lightning server
pip install "imagewright[mcp]"         # the imagewright-mcp server
```

| Extra | Installs | For |
|---|---|---|
| `flux` | `mflux>=0.24.0` | FLUX.2 Klein in process, and the FLUX server |
| `openai` | `openai>=1.30.0` | The OpenAI backend |
| `serve` | `fastapi>=0.116.0`, `uvicorn>=0.30.0`, `pydantic>=2.8.0` | Either server |
| `sdxl` | `torch>=2.4.0`, `diffusers>=0.30.0`, `transformers>=4.44.0`, `accelerate>=0.33.0`, `safetensors>=0.4.0`, `huggingface_hub>=0.24.0` | The SDXL server |
| `mcp` | `mcp>=1.23.0,<2` | `imagewright-mcp` and `as_mcp_image()` |

A backend whose extra is missing raises `MissingExtraError`, a subclass of
`ImportError`. Its message names the `pip install` command that fixes it. For
example, `imagewright serve` with only the `serve` extra installed prints:

```text
Error: mflux not installed; the FLUX server needs the 'serve,flux' extras: pip install "imagewright[serve,flux]"
```

## Render one image

If an image server is running on `localhost:8090` or `localhost:8091`, or the
`flux` extra is installed, run:

```bash
imagewright generate "a lighthouse at dusk, oil painting"
```

The command writes `imagewright-YYYYMMDD-HHMMSS.png` in the current directory,
prints its path, and exits. To choose the file name, pass `-o`:

```bash
imagewright generate "a lighthouse at dusk, oil painting" --seed 7 -o lighthouse.png
```

With no server running and the `flux` extra installed, the command prints a
warning that no server was found and loads FLUX.2 Klein in the calling
process. The first in-process run downloads the model weights from Hugging
Face. To keep the model loaded between renders, run a server instead; see
[Run an image server](servers.md).

If no backend is available, the command fails and lists the fixes; see
[Choose a backend](backends.md).

## Render from Python

```python
from imagewright import ImageSynthesizer, resolve_image_config

synth = ImageSynthesizer(resolve_image_config())  # same order as the CLI
image = synth.generate("a lighthouse at dusk", aspect_ratio="3:2", seed=7)
image.save("lighthouse.png")
```

`generate` returns a PIL image. `generate_b64` takes the same arguments and
returns a base64-encoded PNG. The arguments:

| Argument | Default | Meaning |
|---|---|---|
| `prompt` | required | Text description of the image |
| `aspect_ratio` | `"3:2"` | `1:1`, `3:2`, `2:3`, `16:9`, `9:16`, `4:3` or `3:4` |
| `seed` | random | Seed for a reproducible image |
| `steps` | the config's `steps` (4) | Inference steps; the OpenAI backend ignores it |
| `model` | the config's model | Model id for this call only; `mflux-serve` ignores it |
| `size` | none | `"WIDTHxHEIGHT"`. When it parses, it overrides `aspect_ratio` on the FLUX backends; the OpenAI backend ignores it |

The FLUX backends map an aspect ratio to these sizes:

| Aspect ratio | Size |
|---|---|
| `1:1` | 1024x1024 |
| `3:2` | 1536x1024 |
| `2:3` | 1024x1536 |
| `16:9` | 1536x864 |
| `9:16` | 864x1536 |
| `4:3` | 1365x1024 |
| `3:4` | 1024x1365 |

The OpenAI backend maps an aspect ratio to the nearest size the model
accepts: 1024x1024, 1536x1024 or 1024x1536 for `gpt-image-1`, and 1024x1024,
1792x1024 or 1024x1792 for other models such as `dall-e-3`.

## Install for development

imagewright uses Poetry. Development tools and the docs site are Poetry
groups, not extras.

```bash
git clone https://github.com/Flux-Frontiers/imagewright.git
cd imagewright
poetry install --with dev,docs
.venv/bin/pre-commit install
.venv/bin/pytest --cov
.venv/bin/mkdocs build --strict
```

The tests need no extras: the dev group carries the server and MCP stacks,
and the tests fake the model stacks.
