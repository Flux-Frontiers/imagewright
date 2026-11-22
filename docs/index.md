# imagewright

**Image generation over one HTTP contract.**

*Eric G. Suchanek, PhD -- Flux-Frontiers*

imagewright generates images from text prompts. One contract,
`POST /v1/images/generations`, sits between the code that asks for an image
and the model that renders it, so the same call works against a local server,
a server on another machine, or a model loaded in the calling process.

imagewright depends on no knowledge graph package. It holds the image code
that grew inside [gutenberg_kg](https://github.com/Flux-Frontiers/gutenberg_kg)
and [KG_utils](https://github.com/Flux-Frontiers/kg_utils).

## What you can do with it

- **Render an image from the command line.** `imagewright generate` writes
  one PNG from a prompt. See [Install and render an image](getting-started.md).
- **Render from Python.** `ImageSynthesizer` returns a PIL image from any
  backend. See [Choose a backend](backends.md) and the
  [client API](api/client.md).
- **Run a local image server.** `imagewright serve` serves FLUX.2 Klein on
  Apple Silicon or SDXL-Lightning on CUDA, MPS or CPU. See
  [Run an image server](servers.md).
- **Write your own client or server.** The routes, request fields and
  response bodies are on [The HTTP contract](http-contract.md).
- **Give images to an agent.** `imagewright-mcp` returns a generated image
  to Claude Code or any MCP client. See [MCP server](mcp.md).

## Quick start

```bash
pip install "imagewright[serve,flux]"   # Apple Silicon
imagewright serve &                     # FLUX.2 Klein on 127.0.0.1:8090
imagewright generate "a lighthouse at dusk, oil painting"
```

Every command is listed with its options in the [CLI reference](cli.md).

## License

[Elastic License 2.0](https://www.elastic.co/licensing/elastic-license). The
models are downloaded at run time and carry their own licenses: FLUX.2 Klein
4B is Apache-2.0 and SDXL is OpenRAIL++. FLUX.2 Klein 9B is licensed for
non-commercial use only.
