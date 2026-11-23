"""Run an imagewright server under launchd, so it starts by itself (macOS).

A **login** service is a LaunchAgent in ``~/Library/LaunchAgents``: launchd
starts it when the user logs in and needs no root. A **boot** service is a
LaunchDaemon in ``/Library/LaunchDaemons``: it starts at boot, before anyone
logs in, runs as the user who installed it, and needs root to install. A boot
service suits a headless Mac; MLX reaches the GPU from it.

The service runs ``imagewright serve`` from the environment the installing
``imagewright`` lives in, and launchd restarts it if it exits.
"""

from __future__ import annotations

import getpass
import json
import os
import plistlib
import pwd
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

LABELS = {"flux": "local.imagewright", "sdxl": "local.imagewright-sdxl"}
ENV_PREFIX = {"flux": "MFLUX_SERVER", "sdxl": "SDXL_SERVER"}
DEFAULT_PORTS = {"flux": 8090, "sdxl": 8091}
DAEMON_DIR = Path("/Library/LaunchDaemons")

Runner = Callable[..., subprocess.CompletedProcess]


class ServiceError(RuntimeError):
    """The service could not be installed, started or removed."""


@dataclass(frozen=True)
class ServiceSpec:
    """One launchd service running ``imagewright serve``.

    :param backend: ``flux`` or ``sdxl``.
    :param boot: True for a LaunchDaemon that starts at boot, False for a
        LaunchAgent that starts at login.
    :param user: The account the server runs as.
    :param home: That account's home folder.
    :param executable: The ``imagewright`` script to run.
    :param host: Address the server binds.
    :param port: Port the server binds.
    :param idle_unload_s: Seconds idle before FLUX unloads its model; 0 keeps it.
    """

    backend: str
    boot: bool
    user: str
    home: Path
    executable: Path
    host: str
    port: int
    idle_unload_s: int = 0

    @property
    def label(self) -> str:
        """The launchd label, ``local.imagewright`` for FLUX."""
        return LABELS[self.backend]

    @property
    def plist_path(self) -> Path:
        """Where the property list goes: the system daemons or the user's agents."""
        folder = DAEMON_DIR if self.boot else self.home / "Library" / "LaunchAgents"
        return folder / f"{self.label}.plist"

    @property
    def domain(self) -> str:
        """The launchd domain: ``system`` for a daemon, ``gui/<uid>`` for an agent."""
        return "system" if self.boot else f"gui/{pwd.getpwnam(self.user).pw_uid}"

    @property
    def log_path(self) -> Path:
        """The server's log, under the user's ``~/Library/Logs``."""
        name = "imagewright" if self.backend == "flux" else "imagewright-sdxl"
        return self.home / "Library" / "Logs" / f"{name}.log"

    @property
    def health_url(self) -> str:
        """The URL that answers once the server is up, as seen from this Mac."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "127.0.0.1", "localhost") else self.host
        return f"http://{host}:{self.port}/health"


def target_user(boot: bool) -> tuple[str, Path]:
    """The account a new service runs as, and its home folder.

    Under ``sudo`` that is the user who ran sudo, not root.

    :param boot: True when installing a boot service.
    """
    user = (os.environ.get("SUDO_USER") if boot else None) or getpass.getuser()
    return user, Path(pwd.getpwnam(user).pw_dir)


def imagewright_executable() -> Path:
    """The ``imagewright`` script beside the running Python, so the service uses this venv."""
    return Path(sys.executable).with_name("imagewright")


def build_plist(spec: ServiceSpec) -> dict:
    """The launchd property list for ``spec``.

    A daemon gets no ``HOME``, and the Hugging Face model cache lives under it,
    so the list sets ``HOME`` and ``PATH`` itself.

    :param spec: The service.
    """
    prefix = ENV_PREFIX[spec.backend]
    env = {
        "HOME": str(spec.home),
        "PATH": f"{spec.executable.parent}:/opt/homebrew/bin:/usr/bin:/bin",
        f"{prefix}_HOST": spec.host,
        f"{prefix}_PORT": str(spec.port),
    }
    if spec.backend == "flux":
        env["IMAGE_IDLE_UNLOAD_S"] = str(spec.idle_unload_s)
    plist = {
        "Label": spec.label,
        "ProgramArguments": [str(spec.executable), "serve", "--backend", spec.backend],
        "WorkingDirectory": str(spec.home),
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(spec.log_path),
        "StandardErrorPath": str(spec.log_path),
    }
    if spec.boot:
        plist["UserName"] = spec.user
        plist["GroupName"] = "staff"
    return plist


def _check_privileges(spec: ServiceSpec) -> None:
    if spec.boot and os.geteuid() != 0:
        raise ServiceError(
            "a boot service is a LaunchDaemon and needs root: run "
            f"`sudo {spec.executable} service {{install,uninstall}} --boot ...`"
        )
    if not spec.boot and os.geteuid() == 0:
        raise ServiceError("a login service belongs to a user; run it without sudo")


def _bootout(spec: ServiceSpec, run: Runner) -> None:
    """Stop and unload the service if launchd has it, and wait until it is gone."""
    target = f"{spec.domain}/{spec.label}"
    run(["launchctl", "bootout", target], capture_output=True)
    # bootout returns before the service is gone; bootstrapping too soon fails.
    for _ in range(40):
        if run(["launchctl", "print", target], capture_output=True).returncode != 0:
            return
        time.sleep(0.25)


def _installed_port(spec: ServiceSpec) -> int | None:
    """The port the already-installed copy of this service binds, or None.

    :param spec: The service.
    """
    try:
        env = plistlib.loads(spec.plist_path.read_bytes())["EnvironmentVariables"]
        return int(env[f"{ENV_PREFIX[spec.backend]}_PORT"])
    except (OSError, KeyError, ValueError, plistlib.InvalidFileException):
        return None


def _answers(url: str) -> bool:
    """True if ``url`` answers 200 right now.

    :param url: A ``/health`` URL.
    """
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def wait_healthy(url: str, timeout_s: float = 60.0) -> dict:
    """Poll ``url`` until it answers 200 and return its JSON.

    :param url: The server's ``/health`` URL.
    :param timeout_s: How long to wait.
    :raises ServiceError: If it never answers.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
        if time.monotonic() > deadline:
            raise ServiceError(
                f"the service is loaded but {url} did not answer in {timeout_s:.0f} s"
            )
        time.sleep(1)


def install(spec: ServiceSpec, run: Runner = subprocess.run, timeout_s: float = 60.0) -> dict:
    """Write the property list, (re)load the service and wait for its ``/health``.

    An earlier copy of the same service is replaced.

    :param spec: The service.
    :param run: ``subprocess.run`` or a stand-in, for the ``launchctl`` calls.
    :param timeout_s: How long to wait for ``/health``.
    :returns: The ``/health`` JSON.
    :raises ServiceError: If privileges are wrong, ``launchctl`` refuses, or the
        server does not answer.
    """
    _check_privileges(spec)
    if not spec.executable.exists():
        raise ServiceError(f"{spec.executable} does not exist")
    # Check the port before touching the installed service, so a refusal leaves
    # it running. The service's own port is busy with itself; that is a reinstall.
    if _installed_port(spec) != spec.port and _answers(spec.health_url):
        raise ServiceError(
            f"something else already serves {spec.health_url}; stop it first "
            "(gutenberg_kg's LaunchAgent: `make image-agent-uninstall`)"
        )
    _bootout(spec, run)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.plist_path.parent.mkdir(parents=True, exist_ok=True)
    spec.plist_path.write_bytes(plistlib.dumps(build_plist(spec)))
    spec.plist_path.chmod(0o644)
    if spec.boot:
        os.chown(spec.plist_path, 0, 0)
    done = run(
        ["launchctl", "bootstrap", spec.domain, str(spec.plist_path)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise ServiceError(f"launchctl bootstrap failed: {(done.stderr or '').strip()}")
    return wait_healthy(spec.health_url, timeout_s)


def uninstall(spec: ServiceSpec, run: Runner = subprocess.run) -> bool:
    """Stop the service and delete its property list.

    :param spec: The service.
    :param run: ``subprocess.run`` or a stand-in, for the ``launchctl`` calls.
    :returns: True if a property list was removed.
    """
    _check_privileges(spec)
    _bootout(spec, run)
    if spec.plist_path.exists():
        spec.plist_path.unlink()
        return True
    return False
