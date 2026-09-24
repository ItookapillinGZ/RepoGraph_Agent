"""Opt-in real Docker smoke and isolation acceptance tests."""

import os
import tempfile
import unittest
import uuid
from pathlib import Path

from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy
from sandbox.runner import SandboxRunner

RUN_ACCEPTANCE = os.environ.get("REPOGRAPH_RUN_DOCKER_ACCEPTANCE") == "1"


@unittest.skipUnless(RUN_ACCEPTANCE, "set REPOGRAPH_RUN_DOCKER_ACCEPTANCE=1")
class DockerAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = SandboxPolicy(backend="docker", max_timeout_seconds=10)
        doctor = inspect_sandbox(cls.policy)
        if not doctor.sandbox_image_available:
            raise unittest.SkipTest("Docker daemon or sandbox image unavailable")

    def repository(self, script: str):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "probe.py").write_text(script, encoding="utf-8")
        return temporary, root

    def test_smoke_secret_network_and_workspace_write(self) -> None:
        secret_name = "REPOGRAPH_SANDBOX_SECRET"
        old = os.environ.get(secret_name)
        os.environ[secret_name] = "super-secret-test-value"
        script = """
import os, socket
assert os.environ.get('REPOGRAPH_SANDBOX_SECRET') is None
try:
    socket.create_connection(('203.0.113.1', 9), timeout=1)
except OSError:
    pass
else:
    raise AssertionError('outbound network unexpectedly available')
open('workspace-write.txt', 'w', encoding='utf-8').write('ok')
"""
        temporary, root = self.repository(script)
        try:
            result = SandboxRunner(self.policy).run_repository(
                argv=["python", "probe.py"], repository_root=root,
                timeout_seconds=5, purpose="tests",
            )
        finally:
            temporary.cleanup()
            if old is None:
                os.environ.pop(secret_name, None)
            else:
                os.environ[secret_name] = old
        self.assertEqual(result.status, "passed", result.stderr)
        self.assertEqual(result.provenance.network, "none")

    def test_external_timeout_returns_control(self) -> None:
        temporary, root = self.repository("import time; time.sleep(60)\n")
        try:
            result = SandboxRunner(self.policy).run_repository(
                argv=["python", "probe.py"], repository_root=root,
                timeout_seconds=0.2, purpose="tests",
            )
        finally:
            temporary.cleanup()
        self.assertEqual(result.status, "timeout")
        self.assertTrue(result.timed_out)

    def test_host_sentinel_and_outside_write_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            host_root = Path(directory)
            source = host_root / "source"
            source.mkdir()
            token = uuid.uuid4().hex
            sentinel_name = f"host-only-{token}.txt"
            sentinel = host_root / sentinel_name
            sentinel.write_text(token, encoding="utf-8")
            script = f"""
import os
hits = []
for base, directories, files in os.walk('/'):
    directories[:] = [name for name in directories if name not in ('proc', 'sys', 'dev')]
    if {sentinel_name!r} in files:
        hits.append(os.path.join(base, {sentinel_name!r}))
assert not hits, hits
try:
    open('/outside-{token}', 'w', encoding='utf-8').write('bad')
except OSError:
    pass
else:
    raise AssertionError('container root unexpectedly writable')
"""
            (source / "probe.py").write_text(script, encoding="utf-8")
            result = SandboxRunner(self.policy).run_repository(
                argv=["python", "probe.py"], repository_root=source,
                timeout_seconds=5, purpose="tests",
            )
            self.assertEqual(result.status, "passed", result.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), token)
            self.assertEqual(sorted(path.name for path in source.iterdir()), ["probe.py"])

    def test_process_count_is_bounded(self) -> None:
        script = """
import subprocess, sys
children = []
try:
    for _ in range(64):
        try:
            children.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']))
        except OSError:
            break
    assert len(children) < 64, len(children)
finally:
    for child in children:
        child.terminate()
    for child in children:
        try:
            child.wait(timeout=1)
        except Exception:
            child.kill()
"""
        temporary, root = self.repository(script)
        try:
            result = SandboxRunner(
                SandboxPolicy(backend="docker", max_timeout_seconds=10, pids_limit=16)
            ).run_repository(
                argv=["python", "probe.py"], repository_root=root,
                timeout_seconds=8, purpose="tests",
            )
        finally:
            temporary.cleanup()
        self.assertEqual(result.status, "passed", result.stderr)

    def test_memory_overage_is_contained(self) -> None:
        temporary, root = self.repository(
            "payload = bytearray(256 * 1024 * 1024)\nprint(len(payload))\n"
        )
        try:
            result = SandboxRunner(
                SandboxPolicy(backend="docker", max_timeout_seconds=10, memory_limit_mb=64)
            ).run_repository(
                argv=["python", "probe.py"], repository_root=root,
                timeout_seconds=8, purpose="tests",
            )
        finally:
            temporary.cleanup()
        self.assertEqual(result.status, "failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
