# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
  `pip install` command. Core dependencies gain `httpx` and `pillow`.
- **The repository.** Package layout, CI, pre-commit hooks and a command line
  that reports its version. No image generation yet.
