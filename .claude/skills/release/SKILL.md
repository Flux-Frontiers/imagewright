---
name: release
description: Cut an imagewright release -- sync every version surface, promote the changelog, write release notes, run the live model check and the consumer check, tag, and push. Use when releasing imagewright, publishing it to PyPI, or cutting a version.
---

# Release workflow (imagewright)

Releases here are tag-triggered. Pushing a `v*` tag runs
`.github/workflows/release.yml`, which checks that the tag matches
`pyproject.toml` and that `release-notes.md` names the version, builds the
wheel and sdist, creates the GitHub Release from `release-notes.md`, and
publishes the same files to PyPI by trusted publishing (environment `pypi`).
There are no credentials to supply, and there is no undo: PyPI accepts a
version number once. A mistake can be yanked, not replaced.

Publishing the GitHub Release is also what makes Zenodo archive the version
and mint its DOI.

So the work is verification before the tag. No step below is a formality.

Run every repo command with `env -u VIRTUAL_ENV -u POETRY_ACTIVE`: an
inherited virtualenv from another fleet repo retargets `poetry run` silently.

---

## Step 0 -- Check the preconditions

```bash
git checkout main && git pull origin main
git status --porcelain          # must be empty
```

- `## [Unreleased]` in `CHANGELOG.md` has content. If it is empty, there is
  nothing to release; stop.
- CI and the Docs workflow are green on `main`.
- Before the first release only: the items under "First release" at the end
  of this file are done. Zenodo must be enabled before the GitHub Release is
  published, or that release is never archived.

## Step 1 -- Decide the version

Read `version` in `pyproject.toml`. If a bump already landed, adopt it.
Otherwise ask the user for patch, minor or major unless they said. The tag is
`v<version>`.

## Step 2 -- Refresh the lock

Fleet rule: every release relocks onto the latest releases first, and raises
a floor only when the code needs something new (`kgrag_priv/docs/
FLEET_STANDARDS.md`, "Fleet floors mean needs, locks mean tested").

```bash
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry update
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry install --with dev,docs --sync
```

`poetry update` moves every package to the newest version its constraint
allows; a plain `poetry lock` keeps the versions already locked. Read the
lock diff and name what moved in the changelog.

## Step 3 -- Promote the changelog

Replace `## [Unreleased]` with `## [<version>] - <YYYY-MM-DD>` and insert a
fresh, empty `## [Unreleased]` above it.

Use one ASCII hyphen between the version and the date. `fleet_audit.py`
parses this heading strictly.

Read the promoted section as a user would. It must describe what ships, not
the order of work. For example, 0.1.0's section once ended with "No image
generation yet.", a line from the scaffold commit.

## Step 4 -- Sync every version surface

`imagewright.__version__` is read from the installed package metadata
(`importlib.metadata.version`), so there is no version constant in `src/`.
`pyproject.toml` is the one source; everything below copies it.

```bash
rg -n '^version' pyproject.toml
rg -n '^version:|^date-released:' CITATION.cff
rg -n '[0-9]+\.[0-9]+\.[0-9]+' README.md docs/     # read every hit
head -1 release-notes.md
```

- [ ] `pyproject.toml` `version`
- [ ] `CITATION.cff` `version:`
- [ ] `CITATION.cff` `date-released:`, set to the release date. Add the
      field if it is missing.
- [ ] `README.md` version badge
- [ ] `README.md` status line: `Alpha (X.Y.Z)`
- [ ] `README.md` APA citation: `(Version X.Y.Z) [Software]`
- [ ] `README.md` BibTeX: `version   = {X.Y.Z},` and `year`
- [ ] `docs/getting-started.md` and `docs/cli.md`: the
      `imagewright, version X.Y.Z` output of `imagewright --version`
- [ ] `CHANGELOG.md` heading, promoted in Step 3
- [ ] `release-notes.md`, written in Step 5

`fleet_audit.py` checks only pyproject, CITATION `version:`, the badge and
the changelog heading. The APA and BibTeX blocks, `date-released:` and
`release-notes.md` drift without any tool noticing, so read them.

Prose must describe what ships. A new command, MCP tool or public module is
not released until it has a docs page; `tests/test_docs_coverage.py` enforces
the name, not the content, so read the page.

## Step 5 -- Write `release-notes.md`

The workflow feeds this file to `gh release create --notes-file`. A stale
file publishes the previous release's prose under this tag. Generate it from
the promoted changelog section; never edit it by hand:

```python
import pathlib, re
VERSION, DATE = "X.Y.Z", "YYYY-MM-DD"
cl = pathlib.Path("CHANGELOG.md").read_text()
body = re.search(rf"^## \[{re.escape(VERSION)}\] - {DATE}\n(.*?)(?=^## \[|\Z)",
                 cl, re.S | re.M).group(1).strip("\n")
pathlib.Path("release-notes.md").write_text(
    f"# Release Notes -- v{VERSION}\n\n> Released: {DATE}\n\n{body}\n\n"
    "---\n\n_Full changelog: [CHANGELOG.md](CHANGELOG.md)_\n")
```

The `\Z` alternative matters for the first release, which has no older
heading below it. Confirm the first line names this version.

## Step 6 -- Check that the build is green

```bash
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry install --with dev,docs --sync
env -u VIRTUAL_ENV -u POETRY_ACTIVE .venv/bin/pytest -rs --cov
env -u VIRTUAL_ENV -u POETRY_ACTIVE .venv/bin/pre-commit run --all-files
env -u VIRTUAL_ENV -u POETRY_ACTIVE .venv/bin/mkdocs build --strict
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry check --lock
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry build
```

Require: every test passes, no test skips, coverage is above the
`fail_under` floor, and the strict docs build passes. Unzip the wheel's
`METADATA` and confirm each dependency sits under its intended extra and no
dev or docs tool is listed.

## Step 7 -- Run the live check (REQUIRED before any tag)

The tests fake the model stacks, and the fakes cannot see everything. On
2026-11-22 every test passed while a real render after a preload failed with
MLX's "There is no Stream(cpu, 0) in current thread". Run a real model before
every release, on an Apple Silicon Mac.

A FLUX server may already run on port 8090 on this machine. Leave it alone,
run the test server on 8092, and always pass the endpoint explicitly, or
discovery finds the server on 8090 instead.

1. Install the extras:

    ```bash
    env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry install --with dev -E serve -E flux -E mcp
    ```

2. Start a server that preloads and unloads after 60 idle seconds:

    ```bash
    IMAGE_PRELOAD=1 IMAGE_IDLE_UNLOAD_S=60 .venv/bin/imagewright serve --port 8092
    ```

    Wait for `[startup] model ready`.

3. Render through the client:

    ```bash
    .venv/bin/imagewright generate "a lighthouse at dusk" --seed 7 \
        --endpoint http://127.0.0.1:8092 -o /tmp/iw-live-1.png
    ```

    Open the PNG and confirm it is an image of the prompt.

4. Check both response formats directly:

    ```bash
    curl -s http://127.0.0.1:8092/v1/images/generations \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "a red cube", "size": "512x512"}' | head -c 120
    curl -s http://127.0.0.1:8092/v1/images/generations \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "a red cube", "size": "512x512", "response_format": "filepath"}'
    ```

    The first reply has `data[0].b64_json`; the second has `data[0].filepath`,
    and that file opens as a 512x512 PNG.

5. Wait for the idle unload. Poll `curl -s http://127.0.0.1:8092/health`
   until it shows `"loaded": false`; the watcher checks every 15 seconds at
   this setting, so that can take up to 75 seconds. Then render again as in
   step 3 and confirm the image. This is the path that broke on 2026-11-22.

6. Call the MCP tool through a stdio client, through the server:

    ```bash
    IMAGE_ENDPOINT=http://127.0.0.1:8092 .venv/bin/python - <<'PY'
    import asyncio
    import os
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def main() -> None:
        params = StdioServerParameters(
            command=".venv/bin/imagewright-mcp", env=dict(os.environ))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "generate", {"prompt": "a lighthouse", "size": "512x512", "seed": 7})
                print(result.isError, [(c.type, getattr(c, "mimeType", None)) for c in result.content])

    asyncio.run(main())
    PY
    ```

    Pass `env=dict(os.environ)`: with `env=None`, the MCP client hands the
    child only a short list of safe variables, and `IMAGE_ENDPOINT` would not
    reach it. Require `False` and one `image` item.

7. Stop the test server (Ctrl-C) and confirm it shuts down without a
   traceback. Stop it before the next two steps, so two FLUX models are
   never loaded at once.

8. Repeat step 6 with `IMAGE_BACKEND=mflux-local` and no `IMAGE_ENDPOINT`,
   so the tool loads FLUX in its own process.

9. Run the in-process CLI path:

    ```bash
    .venv/bin/imagewright generate "a lighthouse" --backend mflux-local -o /tmp/iw-live-2.png
    ```

10. Run the SDXL server once against real diffusers, on a machine where
    `pip install "imagewright[serve,sdxl]"` works (CUDA, MPS or CPU):

    ```bash
    env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry install --with dev -E serve -E sdxl
    .venv/bin/imagewright serve --backend sdxl --port 8093
    .venv/bin/imagewright generate "a lighthouse" --endpoint http://127.0.0.1:8093 -o /tmp/iw-sdxl.png
    ```

    The first run downloads about 7 GB.

11. Restore the test environment, so later runs prove the extras-free
    install:

    ```bash
    env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry install --with dev --sync
    ```

Any failure blocks the tag.

## Step 8 -- Test the wheel against every consumer (REQUIRED before tagging)

Fleet floors carry no upper bound, so the next fresh install of a consumer
picks this release up. Rule of record: `kgrag_priv/docs/FLEET_STANDARDS.md`,
"SDK releases are tested against every consumer before tagging".

1. List the consumers from the roster, never a `~/repos/*` glob:

    ```bash
    cd ~/repos/kgrag_priv && env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry run python - <<'PY'
    import pathlib, sys
    sys.path.insert(0, "scripts")
    from fleet_audit import FLEET_REPOS
    for repo in sorted(FLEET_REPOS):
        pyproject = pathlib.Path.home() / "repos" / repo / "pyproject.toml"
        if pyproject.exists() and "imagewright" in pyproject.read_text():
            print(repo)
    PY
    ```

    Read each hit: a comment that names imagewright is not a dependency. If
    the list is empty, say so in the release report and skip to Step 9.

2. Per consumer, one at a time (at most two in parallel), with
   `env -u VIRTUAL_ENV -u POETRY_ACTIVE`:
   `poetry install --with dev --all-extras --sync`; full suite with `-rs` as
   the baseline; `poetry run pip install --force-reinstall --no-deps
   <path to the imagewright wheel>`; full suite again; restore with the same
   sync. Re-download `en_core_web_sm` in `gutenberg_kg` after each sync.
3. Require zero regressions, no new skips, every venv restored, and no
   tracked file changed. A regression blocks the tag.
4. Install each consumer extra that pulls in imagewright into a fresh venv
   with the wheel and exercise it once. Suites cannot see packaging gaps
   between repos.

## Step 9 -- Commit and tag

```bash
git add CHANGELOG.md release-notes.md README.md CITATION.cff pyproject.toml \
        poetry.lock docs/getting-started.md docs/cli.md
git commit -F <message file> -- CHANGELOG.md release-notes.md README.md \
        CITATION.cff pyproject.toml poetry.lock docs/getting-started.md docs/cli.md
git tag -a v<version> -m "v<version>"
```

Run `pre-commit run --all-files` before the commit. The tag must point at the
commit that holds every surface above; after any late fix, move the tag.

## Step 10 -- Push the tag (ASK FIRST, always)

The tag push is the irreversible step. Never push it autonomously. Show the
tag, the commit it points at, the sync checklist, the live check result and
the consumer table, and get explicit approval.

```bash
git push origin main
git push origin v<version>
```

Watch the run: tag check, build, GitHub Release, PyPI publish. Then confirm
`pip install imagewright==<version>` in a fresh venv and that the Docs
workflow deployed.

## Step 11 -- After the release

### Correct the Zenodo license (every release)

Zenodo records an Elastic-2.0 repository as `cc-by-4.0` on every release:
GitHub reports Elastic-2.0 as `NOASSERTION`, and Zenodo falls back without a
warning. `.zenodo.json` cannot prevent it, because Elastic-2.0 has no Zenodo
license id. Correct every version record to a custom rights entry, "Elastic
License 2.0" with link `https://www.elastic.co/licensing/elastic-license`.
The procedure, with the API calls, is step 9a of the generic `/release`
command (`~/.claude/commands/release.md`). It needs `$ZENODO_TOKEN` with
`deposit:write` and `deposit:actions`. Verify through the InvenioRDM view
(`Accept: application/vnd.inveniordm.v1+json`); `metadata.license` reads
`null` afterwards, which is expected.

### Add the DOI (first release, then check every release)

After the first GitHub Release is published and Zenodo has archived it:

1. Get the concept DOI from the Zenodo API. Never copy it from the record
   page's "Cite this version" box, which offers the version DOI. To find a
   record id the first time, start from the repository's numeric GitHub id,
   which Zenodo derives rather than a person typing it:

    ```bash
    gh api repos/Flux-Frontiers/imagewright -q .id
    curl -sI "https://zenodo.org/badge/latestdoi/<that id>" | rg -i '^location'
    ```

    The redirect names the latest version's DOI; its last number is a
    record id. Then list every version and read `conceptdoi`:

    ```bash
    curl -sL "https://zenodo.org/api/records?q=conceptrecid:<recid>&all_versions=1" \
      | python3 -c 'import json,sys; h=json.load(sys.stdin)["hits"]["hits"]; print(h[0]["conceptdoi"], [x["metadata"]["version"] for x in h])'
    ```

    Confirm the newest version listed is the one just released.

2. In one follow-up commit (`docs: add the Zenodo concept DOI`):
   - README: a shields.io DOI badge after the Docs badge,
     `[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.<id>-blue.svg)](https://doi.org/10.5281/zenodo.<id>)`.
     Not a zenodo.org-hosted badge, which often fails to render through
     GitHub's image proxy.
   - `CITATION.cff`: `doi: "10.5281/zenodo.<id>"`.
   - README APA citation: append `https://doi.org/10.5281/zenodo.<id>`.
   - README BibTeX: `doi       = {10.5281/zenodo.<id>},`.
   - README: remove the sentence that says the DOI arrives with the first
     archived release.

At every later release, compare the concept DOI the API returns with the one
the README and `CITATION.cff` declare.

### Downstream

List the consumers whose locks now trail
(`kgrag_priv/scripts/versions_snapshot.py`). A consumer raises its floor only
when it starts using something this release added.

---

## First release

These are one-time settings outside the repository. Each is the owner's to
make; confirm them before Step 10.

- **Repository visibility.** The repository is public.
- **PyPI trusted publisher.** On PyPI, a pending publisher for project
  `imagewright`: owner `Flux-Frontiers`, repository `imagewright`, workflow
  `release.yml`, environment `pypi`. Rename any of the three and the upload
  fails.
- **GitHub environment.** An environment named `pypi` exists under Settings
  -> Environments (GitHub creates it on first use if it is missing; add
  protection rules if wanted).
- **Zenodo.** The repository is switched on at zenodo.org -> GitHub before
  the GitHub Release is published. A release published before that is never
  archived.
- **GitHub Pages.** Settings -> Pages -> Source is "GitHub Actions", or the
  Docs workflow's deploy job fails.
- **Fleet roster.** `imagewright` is added to `FLEET_REPOS` and
  `FLEET_MODULES` in `kgrag_priv/scripts/fleet_audit.py`, so the audit and
  Step 8 see it.
