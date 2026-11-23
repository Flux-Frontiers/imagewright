# Release Notes -- v0.2.0

> Released: 2026-11-23

### Added

- **A logo.** `scripts/build_logo.py` draws imagewright's Art Techno mark (prompt
  lines into the die, a lighthouse whose lamp room is the die), the banner, the
  64 px icon, and the docs site's logo and favicon, which `mkdocs.yml` now uses.
  The README opens with the banner.

- **`imagewright service install` and `uninstall`.** On macOS they run the
  server under launchd: a LaunchAgent that starts at login, or with `--boot`
  (under `sudo`) a LaunchDaemon that starts at boot with nobody logged in, for
  a headless Mac. The service runs `imagewright serve` from the installing
  venv, sets `HOME` for the model cache, waits for `/health`, and refuses to
  install while another server holds the port. `--dry-run` prints the property
  list.

### Changed

- **Beta.** The PyPI classifier is now Development Status 4 - Beta. The
  README drops its status table for a What's new section.

- **The models, by name.** The README and the docs home page have a Models
  section: FLUX.2 Klein 4B (`mlx-community/flux2-klein-4b-4bit`, mflux on
  MLX, Apple Silicon only, port 8090, 4 steps), SDXL-Lightning
  (`ByteDance/SDXL-Lightning` UNets on
  `stabilityai/stable-diffusion-xl-base-1.0`, diffusers on CUDA, MPS or CPU,
  port 8091, about 7 GB on first run) and OpenAI `gpt-image-1`. On a Mac, run
  FLUX; anywhere else, run SDXL.

- **A Zenodo DOI.** The concept DOI `10.5281/zenodo.24450350` is in the README
  (badge, APA and BibTeX) and in `CITATION.cff`. It always resolves to the
  latest archived version.

- **Lock refreshed** onto the latest releases: openai 3.58.0 (the `openai`
  extra), charset-normalizer 3.5.3 and filelock 4.3.1, and in the dev tools
  coverage 7.17.5, python-discovery 1.9.7 and virtualenv 21.15.13. No floor
  changed.

### Security

- **`ImageConfig` no longer prints its API key.** `api_key` is left out of the
  dataclass `repr()`, so printing or logging a config, as `print(cfg)` or an
  f-string does, cannot leak the OpenAI key. A test checks it.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
