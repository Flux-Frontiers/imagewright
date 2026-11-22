# imagewright project instructions

## Overview

imagewright is the fleet's image generation package: a client for
`/v1/images/generations`, local FLUX.2 Klein and SDXL-Lightning servers that
answer that contract, an in-process mode with no server, and an MCP tool. It
is not a KG package and must never import one. `gutenberg_kg` and `kgrag`
depend on it, not the other way round.

The repository is private until its first release. The plan and the order of
work are in `kgrag_priv/docs/KGRAG_IMAGINE_PLAN.md` (sweep item 60).

Fleet-wide rules (dependency conventions, hooks, releases) are in
`kgrag_priv/docs/FLEET_STANDARDS.md`. Cross-repo TODO items go in
`kgrag_priv/FLEET_SWEEP_PLAN.md`, never here.

## Development workflow

```bash
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry install --with dev --all-extras --sync
env -u VIRTUAL_ENV -u POETRY_ACTIVE .venv/bin/pytest --cov
env -u VIRTUAL_ENV -u POETRY_ACTIVE .venv/bin/pre-commit run --all-files
```

Always unset `VIRTUAL_ENV` for repo commands; an inherited venv from another
fleet repo hijacks `poetry run` silently. Run pre-commit on all files before
any commit: the hook chain runs `ty` and the full suite with the coverage
floor.

## Rules for this package

- The HTTP contract is fixed by its consumers: `/health`, `/v1/models` and
  `/v1/images/generations`, with the request fields and both response formats
  `gutenberg_kg`'s servers have today. A change to it is a breaking change.
- Extras are user-facing features: `flux` (mflux, Apple Silicon), `openai`,
  `sdxl` (torch and diffusers) and `mcp`. A core install must import and run
  the client without any of them, and a command that needs an extra must name
  it in its error (`MissingExtraError`), not raise a bare ImportError.
- Never import a model stack (mflux, mlx, openai, torch, diffusers) at
  module scope outside the module that owns it.
- No test may download a model or call a paid API. Fake the model object, and
  run the real FastAPI app and the real client around it.
- A server binds `127.0.0.1` unless told otherwise; it has no authentication.

## Code style

- `:param:` docstrings
- ruff for format and lint (line length 100), ty for types
- plain ASCII in prose, comments and docstrings: `--` not an em dash, `->`
  not an arrow glyph
