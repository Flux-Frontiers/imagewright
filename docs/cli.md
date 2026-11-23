# Every imagewright command

The package installs two commands: `imagewright`, with the subcommands
below, and `imagewright-mcp`, the MCP server described on
[MCP server](mcp.md).

```text
Usage: imagewright [OPTIONS] COMMAND [ARGS]...

  Generate images through a local server, an in-process model or OpenAI.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  generate  Generate an image from PROMPT, write it as a PNG and print...
  serve     Run an image server that answers /v1/images/generations.
```

## imagewright generate

```text
imagewright generate [OPTIONS] PROMPT
```

Generates an image from `PROMPT`, writes it as a PNG and prints its path.

| Option | Default | Meaning |
|---|---|---|
| `--size WxH` | 3:2, 1536x1024 | Output size, for example `1536x1024`. A value that does not parse fails with `Invalid value for --size` |
| `--seed INTEGER` | random | Random seed for a reproducible image |
| `--steps INTEGER` | `IMAGE_STEPS`, else 4 | Inference steps for the FLUX backends |
| `--backend NAME` | chosen as on [Choose a backend](backends.md) | `mflux-local`, `mflux-serve` or `openai` |
| `--endpoint URL` | none | Image server base URL. Implies `mflux-serve` |
| `-o`, `--output FILE` | `imagewright-YYYYMMDD-HHMMSS.png` in the current directory | PNG path to write |

The OpenAI backend ignores `--size` and `--steps`.

The command exits with status 1 when no backend is available, when the
backend's extra is missing, or when the server fails. For example, with
nothing listening on the endpoint:

```bash
imagewright generate "a lighthouse" --endpoint http://127.0.0.1:9
```

```text
Error: image server http://127.0.0.1:9 failed: [Errno 61] Connection refused
```

The error text after `failed:` comes from the operating system and differs by
platform.

## imagewright serve

```text
imagewright serve [OPTIONS]
```

Runs an image server that answers `/v1/images/generations`. The FLUX server
needs the `serve` and `flux` extras; the SDXL server needs `serve` and `sdxl`.

| Option | Default | Meaning |
|---|---|---|
| `--backend [flux\|sdxl]` | `flux` | `flux`: FLUX.2 Klein through mflux (Apple Silicon). `sdxl`: SDXL-Lightning through diffusers (CUDA, MPS or CPU) |
| `--host TEXT` | `MFLUX_SERVER_HOST` (`SDXL_SERVER_HOST` for `sdxl`), else `127.0.0.1` | Address to bind |
| `--port INTEGER` | `MFLUX_SERVER_PORT` (`SDXL_SERVER_PORT` for `sdxl`), else 8090 (8091 for `sdxl`) | Port to bind, 0-65535 |

The server's environment variables, preload, idle unload and bind address are
on [Run an image server](servers.md).

## imagewright service install

```text
imagewright service install [OPTIONS]
```

Installs the server as a launchd service on macOS, starts it and waits for its
`/health`. Without `--boot` it is a LaunchAgent that starts at login; with
`--boot` it is a LaunchDaemon that starts at boot with nobody logged in, and
the command needs `sudo`. The service runs `imagewright serve` from the same
environment as the `imagewright` you run. Installing again replaces it.

| Option | Default | Meaning |
|---|---|---|
| `--backend [flux\|sdxl]` | `flux` | Which server the service runs |
| `--boot` | off | A LaunchDaemon in `/Library/LaunchDaemons` instead of a LaunchAgent |
| `--host TEXT` | `127.0.0.1` | Address to bind; `0.0.0.0` serves the LAN |
| `--port INTEGER` | 8090 (8091 for `sdxl`) | Port to bind |
| `--idle-unload SECONDS` | 0 | FLUX unloads its model after this many idle seconds; 0 keeps it |
| `--dry-run` | off | Print the property list and change nothing |

## imagewright service uninstall

```text
imagewright service uninstall [--backend flux|sdxl] [--boot]
```

Stops the service and removes its property list. Use `--boot` (and `sudo`) for
the LaunchDaemon. When and how to use either is on
[Run an image server](servers.md#run-the-server-as-a-service).

## imagewright --version

```bash
imagewright --version
```

```text
imagewright, version 0.2.0
```

## imagewright-mcp

```text
imagewright-mcp
```

Runs the MCP server over stdio. It takes no options; configure its backend
with the environment variables on [Choose a backend](backends.md).
