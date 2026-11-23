# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`imagewright service install` and `uninstall`.** On macOS they run the
  server under launchd: a LaunchAgent that starts at login, or with `--boot`
  (under `sudo`) a LaunchDaemon that starts at boot with nobody logged in, for
  a headless Mac. The service runs `imagewright serve` from the installing
  venv, sets `HOME` for the model cache, waits for `/health`, and refuses to
  install while another server holds the port. `--dry-run` prints the property
  list.

- **A Zenodo DOI.** The concept DOI `10.5281/zenodo.24450350` is in the README
  (badge, APA and BibTeX) and in `CITATION.cff`. It always resolves to the
  latest archived version.

## [0.1.0] - 2026-11-22

### Added

- **Documentation site** at
  [flux-frontiers.github.io/imagewright](https://flux-frontiers.github.io/imagewright/):
  getting started, choosing a backend, the servers, the HTTP contract, the MCP
  server, the CLI reference and an API reference. A test fails the suite when
  a command, MCP tool or public module has no page.
- **Release workflow.** A `v*` tag builds the wheel and sdist, creates the
  GitHub Release and publishes to PyPI by trusted publishing.
- **`imagewright-mcp`**, an MCP server over stdio with one `generate` tool
  (`prompt`, `size`, `seed`, `steps`, bounded). It chooses its backend with
  `resolve_image_config` on the first call and keeps it. Needs the new `mcp`
  extra (`mcp>=1.23.0,<2`); without it the command exits naming the pip
  command.
- **`compress_image` and `as_mcp_image`** (`imagewright.mcp_image`, also
  exported from `imagewright`), from gutenberg_kg's `_compress_for_mcp` and
  `_as_mcp_image`. They fit an image under `MCP_IMAGE_MAX_BYTES` (3,750,000)
  and keep downscaling until it fits instead of giving up after two rounds.
  `as_mcp_image` returns the bytes in memory rather than writing a temporary
  file. `compress_image` needs only Pillow.
- **Image servers.** `imagewright serve` runs the FLUX.2 Klein server
  (`imagewright.flux_server`, port 8090) or, with `--backend sdxl`, the
  SDXL-Lightning server (`imagewright.sdxl_server`, port 8091). Both are
  ports of gutenberg_kg's `serve/image_server.py` and `serve/sdxl_server.py`
  and answer the same routes, request fields and response bodies. The FLUX
  server keeps one `ImageSynthesizer` loaded, renders one image at a time, and
  honors `IMAGE_PRELOAD` and `IMAGE_IDLE_UNLOAD_S`. The SDXL server also
  renders one image at a time; gutenberg_kg's let two requests share the
  pipeline. Each module builds its app
  with `create_app()` and imports nothing from fastapi or the model stacks at
  module scope.
- **The servers bind `127.0.0.1` by default**, not `0.0.0.0` as gutenberg_kg's
  do; they have no authentication. `--host`, `MFLUX_SERVER_HOST` or
  `SDXL_SERVER_HOST` override it.
- **Extras** `serve` (fastapi, uvicorn, pydantic) and `sdxl` (torch,
  diffusers, transformers, accelerate, safetensors, huggingface_hub). `serve`
  without them exits with the `pip install` command to run.
- **`ImageSynthesizer.load_model()`** loads the in-process model ahead of the
  first render; the FLUX server uses it to preload.
- `NoImageBackendError` names `imagewright serve`.
- **The client.** `ImageSynthesizer`, `ImageConfig`, `ImageBackend` and
  `image_config_from_env`, copied from `kg_utils.synthesis` with the same
  names and signatures. Backends: `mflux-serve` (HTTP), `mflux-local`
  (FLUX.2 Klein in process) and `openai`.
- **In-process model lifecycle.** `ImageSynthesizer.model_loaded` and
  `unload_model()`, from `gutenberg_kg.image_gen`.
- **Discovery.** `discover_image_endpoint` and `DEFAULT_IMAGE_ENDPOINTS` probe
  `IMAGE_ENDPOINT`, `GUTENKG_IMAGE_ENDPOINT`, then `localhost:8090` and
  `:8091`, on `/health` and then `/v1/models`.
- **Backend selection.** `resolve_image_config` chooses, in order: the named
  backend or endpoint, the environment, a discovered server, in-process FLUX
  with a warning, or `NoImageBackendError` listing the fixes.
- **`imagewright generate PROMPT`** writes one PNG and prints its path.
- **Extras.** `flux` (`mflux>=0.24.0`) and `openai` (`openai>=1.30.0`). A
  backend whose extra is missing raises `MissingExtraError` naming the
  `pip install` command. The core dependencies are `click`, `httpx` and
  `pillow`.
- **The repository.** Package layout, CI, pre-commit hooks and
  `imagewright --version`.
