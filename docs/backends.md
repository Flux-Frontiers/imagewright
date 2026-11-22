# Choose a backend

A backend is what renders the image. imagewright has three:

| Backend | What it does | Needs |
|---|---|---|
| `mflux-serve` | Sends the prompt to an image server over HTTP | A server that answers [the HTTP contract](http-contract.md) |
| `mflux-local` | Renders with FLUX.2 Klein in the calling process, with no server | The `flux` extra (Apple Silicon) |
| `openai` | Sends the prompt to OpenAI `gpt-image-1` | The `openai` extra and an API key |

`imagewright generate`, `imagewright-mcp` and `resolve_image_config()` choose
the backend the same way.

## How the backend is chosen

imagewright tries these in order and uses the first that applies:

1. **The backend and endpoint you name.** On the command line, that is
   `--backend` and `--endpoint`; in Python, the `backend` and `endpoint`
   arguments of `resolve_image_config()`. An endpoint with no backend means
   `mflux-serve`, whatever `IMAGE_BACKEND` says.
2. **The environment.** `IMAGE_BACKEND` and `IMAGE_ENDPOINT` (or
   `GUTENKG_IMAGE_ENDPOINT`), the same way. imagewright uses a configured
   endpoint without probing it.
3. **A server that answers locally.** imagewright probes
   `http://localhost:8090`, then `http://localhost:8091`. A server answers if
   `GET /health` or `GET /v1/models` returns 200 within 1.5 seconds.
4. **FLUX.2 Klein in process**, if the `flux` extra is installed. imagewright
   first prints this warning:

    ```text
    warning: No image server found; loading FLUX in process. The first run downloads the weights.
    ```

5. **Otherwise, an error** that lists the fixes.

A model loads in process only at step 4, or when you name `mflux-local`. If
you name `mflux-serve` with no endpoint, imagewright uses a server it finds at
step 3, else `http://localhost:8090`; it never falls through to in-process
FLUX.

When several servers run on one machine, discovery finds the one on port 8090
first. To reach a server on any other port, pass `--endpoint` or set
`IMAGE_ENDPOINT`.

## When no backend is available

`imagewright generate` exits with status 1 and prints the message of
`NoImageBackendError`:

```text
No image backend is available. Any one of these fixes it:
  - start an image server with `imagewright serve` (localhost:8090), or any
    server on localhost:8090 or localhost:8091 (it is found automatically);
  - point at a running server: set IMAGE_ENDPOINT=http://HOST:PORT
    or pass --endpoint;
  - render in process on Apple Silicon: pip install "imagewright[flux]";
  - use OpenAI: pip install "imagewright[openai]", set OPENAI_API_KEY
    and IMAGE_BACKEND=openai (or pass --backend openai).
```

A named backend whose extra is missing fails with the `pip install` command
to run:

```text
Error: mflux is not installed; this backend needs the 'flux' extra: pip install "imagewright[flux]"
Error: openai is not installed; this backend needs the 'openai' extra: pip install "imagewright[openai]"
```

An unknown backend name fails before any server is probed:

```text
Error: unknown image backend 'bogus'; choose one of: mflux-local, mflux-serve, openai
```

## Configure the client with environment variables

| Variable | Meaning | Default |
|---|---|---|
| `IMAGE_BACKEND` | `mflux-serve`, `mflux-local` or `openai` | chosen as above |
| `IMAGE_ENDPOINT` | Image server base URL | none. Fallback: `GUTENKG_IMAGE_ENDPOINT` |
| `IMAGE_MODEL` | Model id for `mflux-local` (a Hugging Face repo) or `openai`. `mflux-serve` does not send it; the server's model renders | `mlx-community/flux2-klein-4b-4bit` for `mflux-local`, `gpt-image-1` for `openai`. Fallback: `GUTENKG_IMAGE_MODEL` |
| `IMAGE_API_KEY` | OpenAI API key | none. Fallback: `OPENAI_API_KEY` |
| `IMAGE_STEPS` | Inference steps for the FLUX backends | `4` |

`IMAGE_STEPS` must be an integer; any other value raises `ValueError`.

`image_config_from_env()` reads the same variables into an `ImageConfig`
without probing anything. Unlike `resolve_image_config()`, it treats an
unknown `IMAGE_BACKEND` as `mflux-serve` instead of raising.

## Build a configuration by hand

To skip selection, build the configuration yourself:

```python
from imagewright import ImageBackend, ImageConfig, ImageSynthesizer

config = ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url="http://gpu-box:8091")
image = ImageSynthesizer(config).generate("a lighthouse at dusk")
```

| Field | Default | Meaning |
|---|---|---|
| `backend` | `ImageBackend.MFLUX_SERVE` | Which backend renders |
| `server_url` | `""`, which means `http://localhost:8090` for `mflux-serve` | Image server base URL |
| `api_key` | `""`, which means `OPENAI_API_KEY` at call time | OpenAI API key |
| `model` | `""`, which means the backend's default model | Model id |
| `steps` | `4` | Inference steps for the FLUX backends |

## Load and unload the in-process model

For `mflux-local`, an `ImageSynthesizer` keeps the model loaded between
calls. `load_model()` loads it ahead of the first render, `model_loaded`
reports whether it is in memory, and `unload_model()` releases it and returns
`True` if a model was loaded.

MLX ties a model to the thread that loaded it. Each `ImageSynthesizer` runs
the load, every render and the unload on one worker thread of its own, so you
can call it from any thread.
