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

Pre-alpha and private. The client works; the servers and the MCP tool are
next.

| Piece | State | What it does |
|---|---|---|
| Client | Done | Sends a prompt to any server that answers the contract and returns a PIL image. Finds a local server on ports 8090 and 8091 when none is configured |
| In-process mode | Done | Renders with FLUX.2 Klein in the calling process, with no server running. Needs the `flux` extra |
| OpenAI backend | Done | Sends the prompt to `gpt-image-1`. Needs the `openai` extra |
| `imagewright generate` | Done | Writes one PNG from a prompt |
| FLUX server | Planned | Serves FLUX.2 Klein through mflux on Apple Silicon. Keeps the model loaded between requests and can unload it when idle |
| SDXL server | Planned | Serves SDXL-Lightning through diffusers on CUDA, MPS or CPU |
| MCP tool | Planned | Returns a generated image to an MCP client, compressed under the client's size limit |

## Install

The core install talks to an image server. Extras add the other backends.
imagewright is not on PyPI yet; until it is, install from a clone or a built
wheel with the same extras.

```bash
pip install imagewright              # client for a running image server
pip install "imagewright[flux]"      # + FLUX.2 Klein in process (Apple Silicon)
pip install "imagewright[openai]"    # + OpenAI gpt-image-1
```

A backend whose extra is missing raises `MissingExtraError`, whose message
names the `pip install` command that fixes it.

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

## Python API

```python
from imagewright import ImageSynthesizer, resolve_image_config

synth = ImageSynthesizer(resolve_image_config())  # same order as the CLI
image = synth.generate("a lighthouse at dusk", aspect_ratio="3:2", seed=7)
image.save("lighthouse.png")
```

| Name | What it is |
|---|---|
| `ImageSynthesizer(config)` | `generate(prompt, *, aspect_ratio="3:2", seed=None, steps=None, model=None, size=None)` returns a PIL image; `generate_b64(...)` returns a base64 PNG. For `mflux-local`, `model_loaded` and `unload_model()` report and release the in-process model |
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
