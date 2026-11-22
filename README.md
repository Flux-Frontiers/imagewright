# imagewright

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)

**imagewright** -- image generation over one HTTP contract.

*Eric G. Suchanek, PhD -- Flux-Frontiers*

imagewright generates images from text prompts. One contract,
`POST /v1/images/generations`, sits between the code that asks for an image
and the model that renders it, so the same call works against a local server,
a server on another machine, or a model loaded in the calling process.

It holds the image code that grew inside
[gutenberg_kg](https://github.com/Flux-Frontiers/gutenberg_kg) and
[KG_utils](https://github.com/Flux-Frontiers/kg_utils). That code never
depended on a knowledge graph, and neither does this package.

## Status

Pre-alpha and private. The client and both image servers work; the MCP tool
is next.

| Piece | State | What it does |
|---|---|---|
| Client | Done | Sends a prompt to any server that answers the contract and returns a PIL image. Finds a local server on ports 8090 and 8091 when none is configured |
| In-process mode | Done | Renders with FLUX.2 Klein in the calling process, with no server running. Needs the `flux` extra |
| OpenAI backend | Done | Sends the prompt to `gpt-image-1`. Needs the `openai` extra |
| `imagewright generate` | Done | Writes one PNG from a prompt |
| FLUX server | Done | `imagewright serve`. Serves FLUX.2 Klein through mflux on Apple Silicon. Keeps the model loaded between requests and can unload it when idle |
| SDXL server | Done | `imagewright serve --backend sdxl`. Serves SDXL-Lightning through diffusers on CUDA, MPS or CPU |
| MCP tool | Planned | Returns a generated image to an MCP client, compressed under the client's size limit |

## Install

The core install talks to an image server. Extras add the other backends.
imagewright is not on PyPI yet; until it is, install from a clone or a built
wheel with the same extras.

```bash
pip install imagewright                # client for a running image server
pip install "imagewright[flux]"        # + FLUX.2 Klein in process (Apple Silicon)
pip install "imagewright[openai]"      # + OpenAI gpt-image-1
pip install "imagewright[serve,flux]"  # + the FLUX server
pip install "imagewright[serve,sdxl]"  # + the SDXL-Lightning server
```

| Extra | Installs | For |
|---|---|---|
| `flux` | mflux | FLUX.2 Klein in process, and the FLUX server |
| `openai` | openai | The OpenAI backend |
| `serve` | fastapi, uvicorn, pydantic | Either server |
| `sdxl` | torch, diffusers, transformers, accelerate, safetensors, huggingface_hub | The SDXL server |

A backend whose extra is missing raises `MissingExtraError`, whose message
names the `pip install` command that fixes it. `imagewright serve` without
its extras exits with the same message.

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

## Install for development

imagewright uses Poetry. Development tools are a Poetry group, not an extra.

```bash
git clone https://github.com/Flux-Frontiers/imagewright.git
cd imagewright
poetry install --with dev
.venv/bin/pre-commit install
```

The tests need no extras: the dev group carries the server stack (fastapi,
uvicorn, pydantic), and the tests fake the model stacks. Install the `flux`
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
[LICENSE](LICENSE).

The models are downloaded at run time and carry their own licenses.
FLUX.2 Klein 4B is Apache-2.0 and SDXL is OpenRAIL++. FLUX.2 Klein 9B is
licensed for non-commercial use only.

## Citation

See [CITATION.cff](CITATION.cff).
