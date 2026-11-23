"""``imagewright service``: the launchd property list and the install and uninstall flow.

launchctl is replaced by a recorder, so the tests write property lists under a
temporary home and never touch launchd.
"""

import getpass
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from imagewright import flux_server, sdxl_server
from imagewright import service as svc
from imagewright.cli import cli
from tests.conftest import ContractServer, closed_port_url


def _spec(tmp_path: Path, **kw) -> svc.ServiceSpec:
    exe = tmp_path / "venv" / "bin" / "imagewright"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.touch()
    fields = {
        "backend": "flux",
        "boot": False,
        "user": getpass.getuser(),
        "home": tmp_path / "home",
        "executable": exe,
        "host": "127.0.0.1",
        "port": 8090,
        "idle_unload_s": 0,
    } | kw
    return svc.ServiceSpec(**fields)


class FakeLaunchctl:
    """Records launchctl calls; ``bootstrap`` brings the contract server's /health up.

    :param server: The contract server standing in for the image server, or None.
    :param bootstrap_rc: Return code ``bootstrap`` gives.
    """

    def __init__(self, server: ContractServer | None = None, bootstrap_rc: int = 0) -> None:
        self.server = server
        self.bootstrap_rc = bootstrap_rc
        self.calls: list[list[str]] = []

    def __call__(self, args, **_kw) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        rc = 0
        if args[1] == "print":
            rc = 113  # not loaded
        elif args[1] == "bootstrap":
            rc = self.bootstrap_rc
            if rc == 0 and self.server is not None:
                self.server.health_status = 200
        return subprocess.CompletedProcess(args, rc, "", "refused" if rc else "")


def test_default_ports_match_the_servers():
    assert svc.DEFAULT_PORTS == {
        "flux": flux_server.DEFAULT_PORT,
        "sdxl": sdxl_server.DEFAULT_PORT,
    }


def test_login_plist_runs_serve_from_this_venv(tmp_path: Path):
    spec = _spec(tmp_path, host="0.0.0.0", idle_unload_s=300)
    plist = svc.build_plist(spec)
    assert plist["Label"] == "local.imagewright"
    assert plist["ProgramArguments"] == [str(spec.executable), "serve", "--backend", "flux"]
    env = plist["EnvironmentVariables"]
    assert env["HOME"] == str(spec.home)
    assert env["PATH"].startswith(str(spec.executable.parent) + ":")
    assert env["MFLUX_SERVER_HOST"] == "0.0.0.0"
    assert env["MFLUX_SERVER_PORT"] == "8090"
    assert env["IMAGE_IDLE_UNLOAD_S"] == "300"
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True
    assert plist["StandardOutPath"] == str(spec.home / "Library/Logs/imagewright.log")
    assert "UserName" not in plist
    assert spec.plist_path == spec.home / "Library/LaunchAgents/local.imagewright.plist"
    assert spec.domain.startswith("gui/")


def test_boot_plist_is_a_daemon_running_as_the_user(tmp_path: Path):
    spec = _spec(tmp_path, boot=True)
    plist = svc.build_plist(spec)
    assert plist["UserName"] == spec.user
    assert plist["GroupName"] == "staff"
    assert spec.plist_path == Path("/Library/LaunchDaemons/local.imagewright.plist")
    assert spec.domain == "system"


def test_sdxl_plist_uses_its_own_label_and_variables(tmp_path: Path):
    spec = _spec(tmp_path, backend="sdxl", port=8091)
    plist = svc.build_plist(spec)
    env = plist["EnvironmentVariables"]
    assert plist["Label"] == "local.imagewright-sdxl"
    assert env["SDXL_SERVER_PORT"] == "8091"
    assert "IMAGE_IDLE_UNLOAD_S" not in env
    assert spec.log_path.name == "imagewright-sdxl.log"


@pytest.mark.parametrize(
    ("host", "url"),
    [("0.0.0.0", "http://127.0.0.1:8090/health"), ("10.0.0.5", "http://10.0.0.5:8090/health")],
)
def test_health_url_reaches_the_server_from_this_mac(tmp_path: Path, host: str, url: str):
    assert _spec(tmp_path, host=host).health_url == url


def test_install_writes_the_plist_bootstraps_and_waits_for_health(
    tmp_path: Path, contract_server: ContractServer
):
    contract_server.health_status = 503  # down until launchd starts it
    port = int(contract_server.url.rsplit(":", 1)[1])
    spec = _spec(tmp_path, port=port)
    launchctl = FakeLaunchctl(contract_server)
    health = svc.install(spec, run=launchctl, timeout_s=10)
    assert health["status"] == "ok"
    assert plistlib.loads(spec.plist_path.read_bytes()) == svc.build_plist(spec)
    assert spec.log_path.parent.is_dir()
    verbs = [c[1] for c in launchctl.calls]
    assert verbs.index("bootout") < verbs.index("bootstrap")
    assert ["launchctl", "bootstrap", spec.domain, str(spec.plist_path)] in launchctl.calls


def test_install_refuses_when_something_else_holds_the_port(
    tmp_path: Path, contract_server: ContractServer
):
    port = int(contract_server.url.rsplit(":", 1)[1])
    spec = _spec(tmp_path, port=port)
    with pytest.raises(svc.ServiceError, match="something else already serves"):
        svc.install(spec, run=FakeLaunchctl(), timeout_s=1)
    assert not spec.plist_path.exists()


def test_a_refused_install_leaves_the_installed_service_alone(
    tmp_path: Path, contract_server: ContractServer
):
    busy = int(contract_server.url.rsplit(":", 1)[1])
    installed = _spec(tmp_path, port=busy + 1)
    installed.plist_path.parent.mkdir(parents=True)
    installed.plist_path.write_bytes(plistlib.dumps(svc.build_plist(installed)))
    launchctl = FakeLaunchctl()
    with pytest.raises(svc.ServiceError, match="something else already serves"):
        svc.install(_spec(tmp_path, port=busy), run=launchctl, timeout_s=1)
    assert launchctl.calls == []
    assert plistlib.loads(installed.plist_path.read_bytes()) == svc.build_plist(installed)


def test_reinstall_on_its_own_busy_port_replaces_the_service(
    tmp_path: Path, contract_server: ContractServer
):
    port = int(contract_server.url.rsplit(":", 1)[1])
    spec = _spec(tmp_path, port=port, idle_unload_s=60)
    spec.plist_path.parent.mkdir(parents=True)
    spec.plist_path.write_bytes(plistlib.dumps(svc.build_plist(_spec(tmp_path, port=port))))
    health = svc.install(spec, run=FakeLaunchctl(contract_server), timeout_s=5)
    assert health["status"] == "ok"
    env = plistlib.loads(spec.plist_path.read_bytes())["EnvironmentVariables"]
    assert env["IMAGE_IDLE_UNLOAD_S"] == "60"


def test_install_reports_a_bootstrap_refusal(tmp_path: Path):
    port = int(closed_port_url().rsplit(":", 1)[1])
    spec = _spec(tmp_path, port=port)
    with pytest.raises(svc.ServiceError, match="bootstrap failed: refused"):
        svc.install(spec, run=FakeLaunchctl(bootstrap_rc=5), timeout_s=1)


def test_install_times_out_when_the_server_never_answers(tmp_path: Path):
    port = int(closed_port_url().rsplit(":", 1)[1])
    spec = _spec(tmp_path, port=port)
    with pytest.raises(svc.ServiceError, match="did not answer"):
        svc.install(spec, run=FakeLaunchctl(), timeout_s=1)


def test_boot_service_needs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc.os, "geteuid", lambda: 501)
    with pytest.raises(svc.ServiceError, match="needs root"):
        svc.install(_spec(tmp_path, boot=True), run=FakeLaunchctl())


def test_login_service_refuses_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svc.os, "geteuid", lambda: 0)
    with pytest.raises(svc.ServiceError, match="without sudo"):
        svc.uninstall(_spec(tmp_path), run=FakeLaunchctl())


def test_uninstall_removes_the_plist(tmp_path: Path):
    spec = _spec(tmp_path)
    spec.plist_path.parent.mkdir(parents=True)
    spec.plist_path.write_bytes(plistlib.dumps(svc.build_plist(spec)))
    launchctl = FakeLaunchctl()
    assert svc.uninstall(spec, run=launchctl) is True
    assert not spec.plist_path.exists()
    assert ["launchctl", "bootout", f"{spec.domain}/{spec.label}"] in launchctl.calls
    assert svc.uninstall(spec, run=launchctl) is False


def test_target_user_under_sudo_is_the_invoking_user(monkeypatch: pytest.MonkeyPatch):
    me = getpass.getuser()
    monkeypatch.setenv("SUDO_USER", me)
    user, home = svc.target_user(boot=True)
    assert user == me and home.is_dir()


def test_cli_dry_run_prints_the_plist_and_changes_nothing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "darwin")  # CI runs on Linux; nothing reaches launchd
    result = CliRunner().invoke(
        cli, ["service", "install", "--dry-run", "--host", "0.0.0.0", "--idle-unload", "300"]
    )
    assert result.exit_code == 0, result.output
    head, body = result.output.split("\n", 1)
    assert head.endswith("Library/LaunchAgents/local.imagewright.plist")
    plist = plistlib.loads(body.encode())
    assert plist["EnvironmentVariables"]["IMAGE_IDLE_UNLOAD_S"] == "300"


def test_cli_service_refuses_other_platforms(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    result = CliRunner().invoke(cli, ["service", "install", "--dry-run"])
    assert result.exit_code != 0
    assert "macOS only" in result.output
