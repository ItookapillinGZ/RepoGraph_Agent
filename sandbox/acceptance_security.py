"""Bounded real-container security and resource probes for H3A."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sandbox.docker import DockerSandboxBackend
from sandbox.models import SandboxExecutionRequest, SandboxExecutionResult
from sandbox.policy import SandboxPolicy
from sandbox.runner import SandboxRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "sandbox" / "acceptance" / "results"


def execution_record(result: SandboxExecutionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "output_truncated": result.output_truncated,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "duration_seconds": round(result.duration_seconds, 6),
        "error_kind": result.error_kind,
        "provenance": result.provenance.model_dump(mode="json"),
    }


def hash_tree(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    import hashlib

    roots = [
        PROJECT_ROOT / "sandbox",
        PROJECT_ROOT / "docker",
        PROJECT_ROOT / "repository_test_tool.py",
        PROJECT_ROOT / "plan_execution.py",
        PROJECT_ROOT / "evaluation" / "evaluator.py",
        PROJECT_ROOT / "test_execution.py",
        PROJECT_ROOT / "tests" / "test_sandbox_acceptance.py",
        PROJECT_ROOT / "tests" / "test_sandbox_docker_command.py",
    ]
    output: dict[str, str] = {}
    for root in roots:
        paths = root.rglob("*") if root.is_dir() else [root]
        for path in paths:
            if (
                not path.is_file()
                or path.suffix == ".pyc"
                or "__pycache__" in path.parts
                or RESULTS_ROOT in path.parents
            ):
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            output[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(output.items()))


def run_command(argv: list[str], timeout: float = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(  # nosec B603
            argv, cwd=PROJECT_ROOT, shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "returncode": None, "error": type(error).__name__}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[:20_000],
        "stderr": completed.stderr[:20_000],
    }


def run_json(argv: list[str]) -> dict[str, Any]:
    completed = run_command(argv)
    if not completed["ok"]:
        return {"ok": False, "error": completed.get("stderr") or completed.get("error")}
    try:
        return {"ok": True, "value": json.loads(completed["stdout"])}
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_json"}


def _probe(script: str, policy: SandboxPolicy, timeout: float = 8) -> SandboxExecutionResult:
    with tempfile.TemporaryDirectory(prefix="repograph-h3a-probe-") as directory:
        root = Path(directory)
        (root / "probe.py").write_text(script, encoding="utf-8")
        return SandboxRunner(policy).run_repository(
            argv=["python", "probe.py"], repository_root=root,
            timeout_seconds=timeout, purpose="tests",
        )


def _argv_evidence() -> dict[str, Any]:
    policy = SandboxPolicy(
        backend="docker", memory_limit_mb=64, cpu_limit=0.5,
        pids_limit=16, max_timeout_seconds=10,
    )
    with tempfile.TemporaryDirectory(prefix="repograph-h3a-argv-") as directory:
        base = Path(directory).resolve()
        workspace = base / "workspace"
        workspace.mkdir()
        backend = DockerSandboxBackend(policy, allowed_workspace_root=base)
        request = SandboxExecutionRequest(
            argv=["python", "probe.py"], workspace=str(workspace),
            timeout_seconds=8, purpose="tests",
        )
        argv = backend._container_argv(request, workspace, "repograph-sbx-redacted")
    rendered = " ".join(argv)
    required = {
        "--rm": "--rm" in argv,
        "--network none": "--network none" in rendered,
        "--cap-drop ALL": "--cap-drop ALL" in rendered,
        "--security-opt no-new-privileges": (
            "--security-opt no-new-privileges" in rendered
        ),
        "--read-only": "--read-only" in argv,
        "--pids-limit 16": "--pids-limit 16" in rendered,
        "--memory 64m": "--memory 64m" in rendered,
        "--cpus 0.5": "--cpus 0.5" in rendered,
        "--tmpfs hardened": "/tmp:rw,noexec,nosuid,size=64m" in argv,  # nosec B108
        "single mount": argv.count("--mount") == 1,
        "no docker socket": "docker.sock" not in rendered,
    }
    return {"passed": all(required.values()), "required": required}


def run_security_and_resources() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = SandboxPolicy(
        backend="docker", max_timeout_seconds=10, memory_limit_mb=64,
        cpu_limit=0.5, pids_limit=16, max_output_chars=1_024,
    )
    secret_name = "REPOGRAPH_SANDBOX_TEST_SECRET"  # nosec B105
    previous = os.environ.get(secret_name)
    os.environ[secret_name] = f"h3a-{uuid.uuid4().hex}"
    boundary_script = r"""
import json, os, pwd, socket
def status_value(name):
    with open('/proc/self/status', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith(name + ':'):
                return line.split(':', 1)[1].strip()
    raise AssertionError(name)
def blocked(path):
    try:
        open(path, 'w', encoding='utf-8').write('blocked')
    except OSError:
        return True
    return False
try:
    socket.create_connection(('203.0.113.1', 9), timeout=1)
except OSError:
    network_blocked = True
else:
    network_blocked = False
assert os.environ.get('REPOGRAPH_SANDBOX_TEST_SECRET') is None
assert network_blocked and os.getuid() == 10001
assert not os.path.exists('/var/run/docker.sock')
assert status_value('NoNewPrivs') == '1'
assert int(status_value('CapEff'), 16) == 0
assert all(blocked(path) for path in (
    '/repograph-root-write', '/etc/repograph-write', '/usr/repograph-write'
))
open('/workspace/workspace-write.txt', 'w', encoding='utf-8').write('ok')
open('/tmp/tmp-write.txt', 'w', encoding='utf-8').write('ok')
pids_max = open('/sys/fs/cgroup/pids.max', encoding='utf-8').read().strip()
memory_max = open('/sys/fs/cgroup/memory.max', encoding='utf-8').read().strip()
cpu_max = open('/sys/fs/cgroup/cpu.max', encoding='utf-8').read().strip()
assert pids_max == '16'
assert int(memory_max) == 64 * 1024 * 1024
quota, period = (int(value) for value in cpu_max.split())
assert abs((quota / period) - 0.5) < 0.001
print(json.dumps({
    'network_blocked': True, 'secret_visible': False, 'uid': os.getuid(),
    'gid': os.getgid(), 'username': pwd.getpwuid(os.getuid()).pw_name,
    'docker_socket_visible': False, 'no_new_privileges': True,
    'effective_capabilities': 0, 'rootfs_write_blocked': True,
    'workspace_writable': True, 'tmp_writable': True,
    'pids_max': int(pids_max), 'memory_max_bytes': int(memory_max),
    'cpu_quota': quota, 'cpu_period': period,
}))
"""
    try:
        boundary = _probe(boundary_script, policy)
    finally:
        if previous is None:
            os.environ.pop(secret_name, None)
        else:
            os.environ[secret_name] = previous
    boundary_evidence: dict[str, Any] = {}
    if boundary.status == "passed":
        try:
            boundary_evidence = json.loads(boundary.stdout)
        except json.JSONDecodeError:
            boundary_evidence = {"parse_error": True}

    with tempfile.TemporaryDirectory(prefix="repograph-h3a-host-") as directory:
        parent = Path(directory)
        source = parent / "source"
        source.mkdir()
        sentinel_name = f"host-only-{uuid.uuid4().hex}.txt"
        sentinel = parent / sentinel_name
        sentinel.write_text("unchanged", encoding="utf-8")
        (source / "probe.py").write_text(
            "import os\n" + f"name = {sentinel_name!r}\n"
            "hits=[]\nfor base, directories, files in os.walk('/'):\n"
            "    directories[:] = [x for x in directories if x not in ('proc','sys','dev')]\n"
            "    if name in files: hits.append(os.path.join(base, name))\n"
            "assert not hits\nopen('disposable-only.txt','w').write('ok')\n",
            encoding="utf-8",
        )
        source_before = hash_tree(source)
        host_isolation = SandboxRunner(policy).run_repository(
            argv=["python", "probe.py"], repository_root=source,
            timeout_seconds=8, purpose="tests",
        )
        source_unchanged = (
            hash_tree(source) == source_before
            and sorted(item.name for item in source.iterdir()) == ["probe.py"]
        )
        sentinel_unchanged = (
            sentinel.exists() and sentinel.read_text(encoding="utf-8") == "unchanged"
        )

    pid_result = _probe(
        "import subprocess,sys\nchildren=[]\ntry:\n"
        "    for _ in range(64):\n"
        "        try: children.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']))\n"
        "        except OSError: break\n"
        "    assert len(children) < 64\n"
        "finally:\n"
        "    for child in children: child.terminate()\n"
        "    for child in children:\n"
        "        try: child.wait(timeout=1)\n"
        "        except Exception: child.kill()\n",
        policy,
    )
    memory_result = _probe(
        "payload=bytearray(256*1024*1024)\nprint(len(payload))\n", policy
    )
    timeout_result = _probe("import time;time.sleep(60)\n", policy, timeout=1)
    output_result = _probe(
        "import sys\nsys.stdout.write('x'*5000)\nsys.stderr.write('y'*5000)\n",
        policy,
    )
    security = {
        "docker_argv": _argv_evidence(),
        "boundary": {
            "passed": boundary.status == "passed" and not boundary_evidence.get("parse_error"),
            "execution": execution_record(boundary), "evidence": boundary_evidence,
        },
        "host_filesystem": {
            "passed": host_isolation.status == "passed"
            and sentinel_unchanged and source_unchanged,
            "sentinel_unchanged": sentinel_unchanged,
            "source_unchanged": source_unchanged,
            "execution": execution_record(host_isolation),
        },
    }
    resources = {
        "pid_limit": {
            "passed": pid_result.status == "passed",
            "execution": execution_record(pid_result),
        },
        "memory_limit": {
            "passed": memory_result.status == "failed",
            "execution": execution_record(memory_result),
        },
        "cpu_limit": {
            "passed": boundary_evidence.get("cpu_quota") == 50_000
            and boundary_evidence.get("cpu_period") == 100_000,
            "quota": boundary_evidence.get("cpu_quota"),
            "period": boundary_evidence.get("cpu_period"),
        },
        "timeout": {
            "passed": timeout_result.status == "timeout" and timeout_result.timed_out,
            "execution": execution_record(timeout_result),
        },
        "output_bounding": {
            "passed": output_result.status == "passed"
            and output_result.output_truncated
            and output_result.stdout_truncated
            and output_result.stderr_truncated
            and len(output_result.stdout) == 1_024
            and len(output_result.stderr) == 1_024,
            "stored_stdout_chars": len(output_result.stdout),
            "stored_stderr_chars": len(output_result.stderr),
            "execution": execution_record(output_result),
        },
    }
    return security, resources


def leak_check() -> dict[str, Any]:
    result = run_command([
        "docker", "ps", "-a", "--filter", "name=repograph-sbx-",
        "--format", "{{.ID}} {{.Names}} {{.Status}}",
    ])
    containers = [
        line for line in result.get("stdout", "").splitlines() if line.strip()
    ]
    return {
        "passed": result["ok"] and not containers,
        "active_or_stopped_repograph_containers": containers,
        "count": len(containers),
    }
