"""Every ``imagewright`` command, MCP tool and public module has to appear on the docs site.

``mkdocs build --strict`` checks links between pages that exist, not pages
that should. This test, copied from vault_kg's, covers that gap: a new
command, tool or module with no page fails the suite.

The rule is deliberately weak -- the name has to appear somewhere in
``docs/`` -- because a strong one (every option, every example) would be
guesswork about what a page should say. A name that appears nowhere is a
feature the site does not admit exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import pytest

from imagewright import mcp_server
from imagewright.cli import cli

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"


def _pages() -> list[Path]:
    """Every Markdown page mkdocs renders, API reference stubs included."""
    return sorted(DOCS.rglob("*.md"))


def _commands(group: click.Group, prefix: str) -> list[str]:
    """Full command names, subcommands of a group included.

    :param group: The click group to walk.
    :param prefix: The command line so far, for example ``"imagewright"``.
    :returns: Names such as ``"imagewright serve"``.
    """
    names: list[str] = []
    for name, cmd in sorted(group.commands.items()):
        names.append(f"{prefix} {name}")
        if isinstance(cmd, click.Group):
            names.extend(_commands(cmd, f"{prefix} {name}"))
    return names


def _tools() -> list[str]:
    """The MCP server's tool names, read from the FastMCP server itself."""
    return [tool.name for tool in asyncio.run(mcp_server.build_server().list_tools())]


@pytest.fixture(scope="module")
def site_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _pages())


def test_there_are_commands_and_tools_to_check() -> None:
    """Guard the guard: an empty registry would make every assertion below pass."""
    assert len(_commands(cli, "imagewright")) >= 2
    assert len(_tools()) >= 1


def test_every_command_appears_on_the_site(site_text: str) -> None:
    missing = [name for name in _commands(cli, "imagewright") if name not in site_text]
    assert not missing, (
        f"not documented anywhere in docs/: {', '.join(missing)}. Add them to a page "
        "listed in mkdocs.yml's nav, or the site will not admit the commands exist."
    )


def test_every_console_script_appears_on_the_site(site_text: str) -> None:
    assert "imagewright-mcp" in site_text


def test_every_mcp_tool_appears_on_the_site(site_text: str) -> None:
    missing = [name for name in _tools() if f"`{name}(" not in site_text]
    assert not missing, f"MCP tools not documented in docs/: {', '.join(missing)}"


def test_every_page_is_in_the_nav() -> None:
    """An unreferenced page builds but is reachable only by guessing its URL.

    Matched as text rather than parsed as YAML: a nav entry is
    ``- Title: path.md``, so the path appearing verbatim tells a listed page
    from an orphaned one.
    """
    nav = MKDOCS.read_text(encoding="utf-8").split("nav:", 1)[-1]
    orphans = [
        path.relative_to(DOCS).as_posix()
        for path in _pages()
        if path.relative_to(DOCS).as_posix() not in nav
    ]
    assert not orphans, f"not in mkdocs.yml nav: {', '.join(orphans)}"


def test_every_module_has_an_api_page() -> None:
    """A new public module with no API page is documented nowhere on the site."""
    documented = {p.stem for p in (DOCS / "api").glob("*.md")}
    # The CLI and the MCP server have their own prose pages; _serving is private.
    prose_only = {"__init__", "cli", "mcp_server", "_serving"}
    modules = {p.stem for p in (ROOT / "src" / "imagewright").glob("*.py")} - prose_only
    assert modules == documented, (
        f"api pages missing for: {sorted(modules - documented)}; "
        f"api pages for no module: {sorted(documented - modules)}"
    )
