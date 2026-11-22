# Run an image server

`imagewright serve` runs an HTTP server that answers
[the HTTP contract](http-contract.md). A server keeps its model loaded between
requests, so only the first render pays for the load. Any client of the
contract can use it: `imagewright generate`, `ImageSynthesizer`,
`imagewright-mcp`, or gutenberg_kg's worker and apps.

| Backend | Model | Runs on | Extras | Default port |
|---|---|---|---|---|
| `flux` (default) | FLUX.2 Klein through mflux | Apple Silicon | `serve,flux` | 8090 |
| `sdxl` | SDXL-Lightning through diffusers | CUDA, MPS or CPU | `serve,sdxl` | 8091 |

## Start a server

```bash
pip install "imagewright[serve,flux]"
imagewright serve                    # FLUX.2 Klein on 127.0.0.1:8090
```

```bash
pip install "imagewright[serve,sdxl]"
imagewright serve --backend sdxl     # SDXL-Lightning on 127.0.0.1:8091
```

Without the extras, the command exits with status 1 and names the
`pip install` command:

```text
Error: torch, diffusers, transformers, accelerate, safetensors, huggingface_hub not installed; the SDXL server needs the 'serve,sdxl' extras: pip install "imagewright[serve,sdxl]"
```

To check that a server is up, ask its health route:

```bash
curl http://127.0.0.1:8090/health
```

Each server renders one image at a time; a second request waits for the
first to finish.

## Choose the bind address

| Option | Default |
|---|---|
| `--host HOST` | `MFLUX_SERVER_HOST` (`SDXL_SERVER_HOST` for `sdxl`), else `127.0.0.1` |
| `--port PORT` | `MFLUX_SERVER_PORT` (`SDXL_SERVER_PORT` for `sdxl`), else `8090` (`8091` for `sdxl`) |

The servers have no authentication: anyone who can reach the port can render
images and fill the output directory. So they bind `127.0.0.1` by default, and
only the machine they run on can reach them. gutenberg_kg's servers bound
`0.0.0.0`. To share a server, pass `--host 0.0.0.0`, and do it only on a
network you trust.

## Run one FLUX server per machine

A FLUX server holds 12-15 GB with the model loaded, and every FLUX server
defaults to port 8090, including the one gutenberg_kg runs. Run one per
machine and point every client at it. To run a second one for a test, give it
another port and pass that endpoint to the client explicitly, or discovery
finds the server on 8090 instead:

```bash
imagewright serve --port 8092
imagewright generate "a lighthouse at dusk" --endpoint http://127.0.0.1:8092
```

## Configure the FLUX server

| Variable | Meaning | Default |
|---|---|---|
| `IMAGE_MODEL` | Hugging Face repo for FLUX.2 Klein. Fallback: `GUTENKG_IMAGE_MODEL` | `mlx-community/flux2-klein-4b-4bit` |
| `IMAGE_STEPS` | Inference steps when a request does not set them, or sets 0 | `4` |
| `IMAGE_PRELOAD` | `1`, `true` or `yes` (any case) loads the model before the server accepts requests | load on the first request |
| `IMAGE_IDLE_UNLOAD_S` | Unload the model after this many idle seconds | `0`, never |
| `IMAGE_OUTPUT_DIR` | Where `response_format=filepath` images go | `imagewright_images` in the system temp directory |
| `MFLUX_SERVER_HOST` | Bind address when `--host` is not given | `127.0.0.1` |
| `MFLUX_SERVER_PORT` | Port when `--port` is not given | `8090` |

Whatever repo `IMAGE_MODEL` names, the server reports its model as
`flux2-klein-4b` on `/health` and `/v1/models`; the contract fixes that id.

### Preload the model

By default the model loads on the first render, so the port opens at once and
the first request waits for the load. With `IMAGE_PRELOAD=1`, the server loads
the model before it accepts requests, and the first request renders at full
speed:

```bash
IMAGE_PRELOAD=1 imagewright serve
```

### Unload the model when idle

With `IMAGE_IDLE_UNLOAD_S` above 0, the server unloads the model after that
many seconds without a render, so a Mac with little memory gets its 12-15 GB
back. The next request loads the model again.

```bash
IMAGE_PRELOAD=1 IMAGE_IDLE_UNLOAD_S=600 imagewright serve
```

The server checks for an idle model every quarter of `IMAGE_IDLE_UNLOAD_S`,
kept between 5 and 30 seconds, so the unload can come up to that much later
than the limit. It never unloads in the middle of a render. `/health` reports
`loaded` and `idle_unload_s`, so you can watch it happen.

## Configure the SDXL server

| Variable | Meaning | Default |
|---|---|---|
| `SDXL_MODEL` | Lightning UNet: `sdxl_lightning_2`, `sdxl_lightning_4` or `sdxl_lightning_8` | `sdxl_lightning_4` |
| `SDXL_BASE` | SDXL base repo | `stabilityai/stable-diffusion-xl-base-1.0` |
| `SDXL_OFFLINE` | `1`, `true` or `yes` never contacts Hugging Face; the weights must be cached | download on first run |
| `MPS_DTYPE` | `float16`, with the fp16-fix VAE, or `float32` | `float16` |
| `IMAGE_OUTPUT_DIR` | Where `response_format=filepath` images go | `imagewright_images` in the system temp directory |
| `SDXL_SERVER_HOST` | Bind address when `--host` is not given | `127.0.0.1` |
| `SDXL_SERVER_PORT` | Port when `--port` is not given | `8091` |

The SDXL server builds its pipeline before it opens its port, so the port
opens only once the server can render. The first run downloads about 7 GB of
weights into the Hugging Face cache.

The server picks the device in this order: CUDA in float16, MPS in the
`MPS_DTYPE` precision, CPU in float32. In float16 it swaps in the
`madebyollin/sdxl-vae-fp16-fix` VAE, because the stock SDXL VAE overflows to
black images in float16.

A Lightning UNet is distilled for a fixed step count, so the server runs 2, 4
or 8 steps for the variant `SDXL_MODEL` names, whatever the request asks.

## Run a server from Python

Each server module builds its FastAPI app with `create_app()`. To configure a
server in code instead of from the environment, pass a server object:

```python
import uvicorn
from imagewright.flux_server import FluxServer, create_app

server = FluxServer(preload=True, idle_unload_s=600)
uvicorn.run(create_app(server), host="127.0.0.1", port=8090)
```

`SdxlServer` takes `model`, `base`, `offline` and `output_dir` the same way.
See the [FLUX server](api/flux_server.md) and [SDXL server](api/sdxl_server.md)
API pages.
