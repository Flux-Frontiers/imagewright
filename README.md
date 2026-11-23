<p align="center">
  <img src="https://raw.githubusercontent.com/Flux-Frontiers/imagewright/main/assets/imagewright_banner.png" alt="imagewright" width="720"/>
</p>

# imagewright

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![PyPI](https://img.shields.io/pypi/v/imagewright.svg)](https://pypi.org/project/imagewright/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/Flux-Frontiers/imagewright/releases)
[![CI](https://github.com/Flux-Frontiers/imagewright/actions/workflows/ci.yml/badge.svg)](https://github.com/Flux-Frontiers/imagewright/actions/workflows/ci.yml)
[![Docs](https://github.com/Flux-Frontiers/imagewright/actions/workflows/docs.yml/badge.svg)](https://flux-frontiers.github.io/imagewright/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24450350-blue.svg)](https://doi.org/10.5281/zenodo.24450350)

**imagewright** -- image generation over one HTTP contract. Beta.

*Eric G. Suchanek, PhD -- Flux-Frontiers*

imagewright generates images from text prompts. One contract,
`POST /v1/images/generations`, sits between the code that asks for an image
and the model that renders it, so the same call works against a local server,
a server on another machine, or a model loaded in the calling process.

It holds the image code that grew inside
[gutenberg_kg](https://github.com/Flux-Frontiers/gutenberg_kg) and
[KG_utils](https://github.com/Flux-Frontiers/kg_utils). That code never
depended on a knowledge graph, and neither does this package.

**Documentation:** [flux-frontiers.github.io/imagewright](https://flux-frontiers.github.io/imagewright/)
covers installation, choosing a backend, the servers, the HTTP contract, the
MCP server, every command and the Python API.

## What's new

**Unreleased**

- **A logo.** imagewright's Art Techno mark: prompt lines run into the die,
  and the picture out is a lighthouse whose lamp room is the die.
  `scripts/build_logo.py` draws the mark, banner, icon and the docs site's
  logo and favicon.
- **`imagewright service install`** runs the image server under launchd on
  macOS, so it starts by itself: at login, or with `--boot` (under `sudo`)
  at boot with nobody logged in, for a headless Mac. `imagewright service
  uninstall` removes it. See [Run the server as a service](#run-the-server-as-a-service).
- **A Zenodo DOI**, `10.5281/zenodo.24450350`, in the badge, the citation and
  `CITATION.cff`.

**0.1.0** (2026-11-22): first release. The client, in-process FLUX, the
OpenAI backend, the FLUX and SDXL servers, `imagewright generate` and the
`imagewright-mcp` tool.

## Install

imagewright is on PyPI and needs Python 3.12 or 3.13. The core install
talks to an image server. Extras add the other backends and the servers.

```bash
pip install imagewright                # client for a running image server
pip install "imagewright[flux]"        # + FLUX.2 Klein in process (Apple Silicon)
pip install "imagewright[openai]"      # + OpenAI gpt-image-1
pip install "imagewright[serve,flux]"  # + the FLUX server
pip install "imagewright[serve,sdxl]"  # + the SDXL-Lightning server
pip install "imagewright[mcp]"         # + the imagewright-mcp server
```

| Extra | Installs | For |
|---|---|---|
| `flux` | mflux | FLUX.2 Klein in process, and the FLUX server |
| `openai` | openai | The OpenAI backend |
| `serve` | fastapi, uvicorn, pydantic | Either server |
| `sdxl` | torch, diffusers, transformers, accelerate, safetensors, huggingface_hub | The SDXL server |
| `mcp` | mcp | `imagewright-mcp` and `as_mcp_image()` |

A backend whose extra is missing raises `MissingExtraError`, whose message
names the `pip install` command that fixes it. `imagewright serve` without
its extras, or `imagewright-mcp` without `mcp`, exits with the same
message.

## Generate an image

```bash
imagewright generate "a lighthouse at dusk, oil painting"
```

The command writes `imagewright-YYYYMMDD-HHMMSS.png` in the current directory
and prints its path. Options:

| Option | Meaning |
|---|---|
| `--size WxH` | Output size, for example `1536x1024`. Default: 3:2, 1536x1024 |
| `--seed N` | Random seed for a reproducible image |
| `--steps N` | Inference steps for the FLUX backends. Default: 4 |
| `--backend NAME` | `mflux-serve`, `mflux-local` or `openai` |
| `--endpoint URL` | Image server base URL. Implies `mflux-serve` |
| `-o PATH` | Where to write the PNG |

### How the backend is chosen

1. `--backend` and `--endpoint`. An endpoint with no backend means
   `mflux-serve`, whatever `IMAGE_BACKEND` says.
2. `IMAGE_BACKEND` and `IMAGE_ENDPOINT` (or `GUTENKG_IMAGE_ENDPOINT`) from
   the environment. A configured endpoint is used without probing it.
3. A server that answers on `http://localhost:8090` or
   `http://localhost:8091`. A server answers if `/health` or `/v1/models`
   returns 200.
4. FLUX.2 Klein in process, if the `flux` extra is installed. A warning says
   that no server was found and the model is loading in process.
5. Otherwise the command fails with a list of the ways to fix it.

A model loads in process only at step 4, or when `mflux-local` is named.

### Environment variables

| Variable | Meaning |
|---|---|
| `IMAGE_BACKEND` | `mflux-serve`, `mflux-local` or `openai` |
| `IMAGE_ENDPOINT` | Image server base URL. Fallback: `GUTENKG_IMAGE_ENDPOINT` |
| `IMAGE_MODEL` | Model id. Fallback: `GUTENKG_IMAGE_MODEL` |
| `IMAGE_API_KEY` | OpenAI key. Fallback: `OPENAI_API_KEY` |
| `IMAGE_STEPS` | Inference steps. Default: 4 |

## Run an image server

```bash
imagewright serve                    # FLUX.2 Klein on 127.0.0.1:8090
imagewright serve --backend sdxl     # SDXL-Lightning on 127.0.0.1:8091
```

Both servers answer the same contract, the one gutenberg_kg's image servers
answer, so gutenberg_kg's worker and apps can point at either one unchanged:

| Route | Answers |
|---|---|
| `GET /health` | `status`, `backend` and `model`; the FLUX server adds `loaded` and `idle_unload_s`. Does no model work, so it answers during a render and before the weights load |
| `GET /v1/models` | The one model the server serves, in OpenAI list format |
| `POST /v1/images/generations` | `prompt` (required), `size` (`WIDTHxHEIGHT`), `seed`, `num_inference_steps`, `response_format`; the SDXL server also takes `negative_prompt`. Returns `{"created": ..., "data": [{"b64_json": ...}]}`, or `{"filepath": ...}` for `response_format=filepath` |

Each server renders one image at a time. A size that does not parse
gives 1536x1024 on FLUX and 1024x1024 on SDXL. The SDXL server runs the
step count its Lightning UNet was trained for, whatever the request asks.

| Option | Meaning |
|---|---|
| `--backend flux\|sdxl` | Which server. Default: `flux` |
| `--host HOST` | Address to bind. Default: `MFLUX_SERVER_HOST` (`SDXL_SERVER_HOST` for sdxl), else `127.0.0.1` |
| `--port PORT` | Port to bind. Default: `MFLUX_SERVER_PORT` (`SDXL_SERVER_PORT` for sdxl), else 8090 (8091 for sdxl) |

The servers have no authentication, so they bind `127.0.0.1` by default.
gutenberg_kg's servers bound `0.0.0.0`; pass `--host 0.0.0.0` to share a
server on a network you trust.

Run one FLUX server per machine. Each holds 12-15 GB with the model loaded,
and every FLUX server defaults to port 8090, including the one gutenberg_kg
runs. To run a second one for a test, give it another port, for example
`--port 8092`.

### Server environment variables

| Variable | Server | Meaning |
|---|---|---|
| `IMAGE_MODEL` | FLUX | HF repo for FLUX.2 Klein. Fallback: `GUTENKG_IMAGE_MODEL`. Default: `mlx-community/flux2-klein-4b-4bit` |
| `IMAGE_STEPS` | FLUX | Steps when a request does not set them. Default: 4 |
| `IMAGE_PRELOAD` | FLUX | `1`, `true` or `yes` loads the model before the server accepts requests. Default: load on the first request |
| `IMAGE_IDLE_UNLOAD_S` | FLUX | Unload the model after this many idle seconds, so a small Mac gets its memory back; the next request reloads it. Default: 0, never |
| `MFLUX_SERVER_HOST`, `MFLUX_SERVER_PORT` | FLUX | Bind address and port, when `--host` and `--port` are not given |
| `SDXL_MODEL` | SDXL | `sdxl_lightning_2`, `sdxl_lightning_4` (default) or `sdxl_lightning_8` |
| `SDXL_BASE` | SDXL | SDXL base repo. Default: `stabilityai/stable-diffusion-xl-base-1.0` |
| `SDXL_OFFLINE` | SDXL | `1` never contacts Hugging Face; the weights must be cached. Default: download them on first run (about 7 GB) |
| `MPS_DTYPE` | SDXL | `float16` (default, with the fp16-fix VAE) or `float32` |
| `SDXL_SERVER_HOST`, `SDXL_SERVER_PORT` | SDXL | Bind address and port, when `--host` and `--port` are not given |
| `IMAGE_OUTPUT_DIR` | Both | Where `response_format=filepath` images go. Default: `imagewright_images` in the system temp directory |

The SDXL server builds its pipeline before it opens its port.

### Run the server as a service

On macOS, launchd can start the server by itself and restart it if it exits:

```bash
imagewright service install --host 0.0.0.0                 # at login
sudo "$(which imagewright)" service install --boot \
    --host 0.0.0.0 --idle-unload 300                        # at boot, no login needed
imagewright service uninstall                              # remove it (--boot and sudo for the daemon)
```

The service runs `imagewright serve` from the environment you install it
from and logs to `~/Library/Logs/imagewright.log`. It refuses to install while
another server answers on its port. Add `--dry-run` to print the launchd
property list without changing anything. With FileVault on, a boot stops at
the unlock screen until someone types the password; `sudo fdesetup
authrestart` gets a planned restart past it.

## Use it from an MCP client

`imagewright-mcp` is an MCP server over stdio with one tool, `generate`:

| Argument | Meaning |
|---|---|
| `prompt` | Description of the image. Required; at most 4000 characters |
| `size` | `WIDTHxHEIGHT`. Default: `1536x1024` |
| `seed` | Integer seed for a reproducible image. Default: random |
| `steps` | Inference steps, 1-50. Default: `IMAGE_STEPS`, else 4 |

The tool chooses its backend on its first call, in the same order as
`imagewright generate`, and keeps it for the life of the server, so an
in-process model loads once. It returns the image as PNG, or as JPEG when the
PNG is over 3.75 MB (5 MB once base64-encoded, the usual MCP client limit).

To add it to Claude Code, put this in the project's `.mcp.json`:

```json
{
  "mcpServers": {
    "imagewright": { "command": "imagewright-mcp" }
  }
}
```

## Python API

```python
from imagewright import ImageSynthesizer, resolve_image_config

synth = ImageSynthesizer(resolve_image_config())  # same order as the CLI
image = synth.generate("a lighthouse at dusk", aspect_ratio="3:2", seed=7)
image.save("lighthouse.png")
```

| Name | What it is |
|---|---|
| `ImageSynthesizer(config)` | `generate(prompt, *, aspect_ratio="3:2", seed=None, steps=None, model=None, size=None)` returns a PIL image; `generate_b64(...)` returns a base64 PNG. For `mflux-local`, `load_model()`, `model_loaded` and `unload_model()` load, report and release the in-process model |
| `ImageConfig`, `ImageBackend` | The backend and its settings |
| `image_config_from_env()` | An `ImageConfig` from the environment variables above |
| `resolve_image_config(backend=None, endpoint=None)` | Chooses the backend in the order above |
| `discover_image_endpoint(candidates=None)` | The first image server that answers, or `None`. By default it probes `IMAGE_ENDPOINT` and `GUTENKG_IMAGE_ENDPOINT` (when set) before the two localhost ports |
| `DEFAULT_IMAGE_ENDPOINTS` | `http://localhost:8090`, `http://localhost:8091` |
| `NoImageBackendError` | Raised when no backend is available; the message lists the fixes |
| `MissingExtraError` | Raised when a backend's extra is not installed; a subclass of `ImportError` |
| `compress_image(image, max_bytes=MCP_IMAGE_MAX_BYTES)` | Encodes a PIL image in at most `max_bytes`: PNG if it fits, else JPEG at quality 85, 70 and 50, else JPEG shrunk by a quarter per round until it fits. Returns `(data, "png" or "jpeg")`. Needs only Pillow |
| `as_mcp_image(image, max_bytes=MCP_IMAGE_MAX_BYTES)` | `compress_image`, wrapped as the `mcp.server.fastmcp.Image` a FastMCP tool returns. Needs the `mcp` extra |
| `MCP_IMAGE_MAX_BYTES` | 3,750,000 |

## Install for development

imagewright uses Poetry. Development tools are a Poetry group, not an extra.

```bash
git clone https://github.com/Flux-Frontiers/imagewright.git
cd imagewright
poetry install --with dev
.venv/bin/pre-commit install
```

The tests need no extras: the dev group carries the server and MCP stacks
(fastapi, uvicorn, pydantic, mcp), and the tests fake the model stacks. Install the `flux`
or `sdxl` extra only to run a real model.

## Run the checks

```bash
.venv/bin/pytest --cov
.venv/bin/pre-commit run --all-files
```

To confirm the install, run the command line:

```bash
.venv/bin/imagewright --version
```

## License

[Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) -- see
[LICENSE](https://github.com/Flux-Frontiers/imagewright/blob/main/LICENSE).

The models are downloaded at run time and carry their own licenses.
FLUX.2 Klein 4B is Apache-2.0 and SDXL is OpenRAIL++. FLUX.2 Klein 9B is
licensed for non-commercial use only.

## Citation

If you use imagewright in your research or project, please cite it:

> Suchanek, E. G. (2026). *imagewright: Image Generation Over One HTTP Contract* (Version 0.1.0) [Software]. Flux-Frontiers. https://doi.org/10.5281/zenodo.24450350

```bibtex
@software{suchanek_imagewright,
  author    = {Suchanek, Eric G.},
  title     = {{imagewright}: Image Generation Over One HTTP Contract},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Flux-Frontiers},
  doi       = {10.5281/zenodo.24450350},
  url       = {https://github.com/Flux-Frontiers/imagewright},
}
```

See the
[changelog](https://github.com/Flux-Frontiers/imagewright/blob/main/CHANGELOG.md)
for release history.
