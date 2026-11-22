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

Pre-alpha and private. The repository has its layout, CI and hooks, and a
command line that reports its version. It does not generate images yet.

| Planned | What it does |
|---|---|
| Client | Sends a prompt to any server that answers the contract and returns a PIL image. Finds a local server on ports 8090 and 8091 when none is configured |
| FLUX server | Serves FLUX.2 Klein through mflux on Apple Silicon. Keeps the model loaded between requests and can unload it when idle |
| SDXL server | Serves SDXL-Lightning through diffusers on CUDA, MPS or CPU |
| In-process mode | Renders with FLUX.2 Klein in the calling process, with no server running |
| OpenAI backend | Sends the prompt to `gpt-image-1` |
| MCP tool | Returns a generated image to an MCP client, compressed under the client's size limit |

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
