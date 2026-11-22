# The HTTP contract

Every imagewright server answers three routes, with the request fields and
response bodies gutenberg_kg's image servers have always used. A client of
gutenberg_kg's servers can point at an imagewright server unchanged, and any
server that answers these routes works with the imagewright client. A change
to this contract is a breaking change.

## GET /health

A liveness probe. It does no model work, so it answers during a render and
before the weights load. Discovery treats a 200 here as "a server is on this
port".

The FLUX server answers:

```json
{"status": "ok", "backend": "flux", "model": "flux2-klein-4b", "loaded": false, "idle_unload_s": 0}
```

`loaded` is `true` while the model is in memory; `idle_unload_s` is the
`IMAGE_IDLE_UNLOAD_S` setting.

The SDXL server answers:

```json
{"status": "ok", "backend": "sdxl-lightning", "model": "sdxl_lightning_4"}
```

## GET /v1/models

The one model the server serves, in OpenAI list format. Discovery falls back
to this route for a server that has no `/health`.

FLUX server:

```json
{"object": "list", "data": [{"id": "flux2-klein-4b", "object": "model", "owned_by": "mflux"}]}
```

SDXL server:

```json
{"object": "list", "data": [{"id": "sdxl_lightning_4", "object": "model", "owned_by": "sdxl-lightning"}]}
```

## POST /v1/images/generations

Renders one image. The request body is JSON:

| Field | Type | FLUX default | SDXL default | Meaning |
|---|---|---|---|---|
| `prompt` | string | required | required | Text description of the image |
| `size` | string | `"1536x1024"` | `"1024x1024"` | `WIDTHxHEIGHT`. A size that does not parse gives the default |
| `seed` | integer or null | random | random | Seed for a reproducible image |
| `num_inference_steps` | integer or null | `IMAGE_STEPS`, else 4 | the Lightning variant's steps | FLUX uses it when above 0. SDXL uses it only for a model that is not a known Lightning variant |
| `response_format` | string | `"b64_json"` | `"b64_json"` | `"filepath"` writes the PNG on the server; any other value returns it inline |
| `negative_prompt` | string | not accepted | `"blurry, bad quality, distorted"` | What to steer away from. SDXL only |
| `model` | string | `"flux2-klein-4b"` | `"sdxl_lightning_4"` | Accepted and ignored |
| `n` | integer | `1` | `1` | Accepted and ignored; the server returns one image |
| `quality` | string or null | null | null | Accepted and ignored |

A request with no `prompt` gets a 422 response from FastAPI's validation.

### Response formats

With `response_format` `"b64_json"`, or any value other than `"filepath"`,
the response carries the PNG as base64:

```json
{"created": 1795392000, "data": [{"b64_json": "iVBORw0KGgo..."}]}
```

With `"filepath"`, the server writes the PNG under `IMAGE_OUTPUT_DIR` with a
random name and returns its absolute path. On Linux with the default output
directory, for example:

```json
{"created": 1795392000, "data": [{"filepath": "/tmp/imagewright_images/<32 hex characters>.png"}]}
```

`created` is the Unix time of the reply. A `filepath` is a path on the
server's machine, so use that format only when the client shares the
server's file system.

## What the client sends

`ImageSynthesizer` with the `mflux-serve` backend posts `prompt`, `n` (1),
`size`, `num_inference_steps`, `response_format` (`"b64_json"`) and, when
given, `seed`. It reads `data[0].b64_json` from the reply. It waits up to
5 seconds to connect and 300 seconds for the image.
