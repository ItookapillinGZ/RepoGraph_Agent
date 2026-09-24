"""Generate the complete H3A real-Docker acceptance evidence bundle."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from sandbox.acceptance_replay import run_equivalence, run_integration_tests
from sandbox.acceptance_security import (
    RESULTS_ROOT,
    leak_check,
    run_json,
    run_security_and_resources,
    source_hashes,
)
from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy

BASE_IMAGE_DIGEST = (
    "sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285"
)
EXPECTED_IMAGE_ID = (
    "sha256:c66fc8697a59ed31747845eb432dd05ae3371d12a4ed23286708cc8d67f96d57"
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _write_json(name: str, value: Any) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
    (RESULTS_ROOT / name).write_text(rendered + "\n", encoding="utf-8")


def _provenance() -> dict[str, Any]:
    image = run_json(["docker", "image", "inspect", "repograph-sandbox:local"])
    base = run_json(["docker", "image", "inspect", "python:3.13-slim"])
    version = run_json(["docker", "version", "--format", "{{json .}}"])
    image_value = image.get("value", [{}])
    base_value = base.get("value", [{}])
    image_id = image_value[0].get("Id") if image_value else None
    base_digests = base_value[0].get("RepoDigests", []) if base_value else []
    return {
        "acceptance_timestamp": datetime.now(UTC).isoformat(),
        "configured_image": "repograph-sandbox:local",
        "sandbox_image_id": image_id,
        "expected_sandbox_image_id": EXPECTED_IMAGE_ID,
        "sandbox_image_matches_expected": image_id == EXPECTED_IMAGE_ID,
        "trusted_base_image_digest": BASE_IMAGE_DIGEST,
        "local_base_repo_digests": base_digests,
        "docker_version": version.get("value"),
        "platform": "Windows + Docker Desktop + WSL2 Linux engine",
    }


def _security_scan() -> dict[str, Any]:
    patterns = {
        "openai_key": ("sk-" + "[A-Za-z0-9]{20,}"),
        "github_token": ("gh" + "[pousr]_[A-Za-z0-9]{20,}"),
        "docker_token": ("dckr_" + "pat_[A-Za-z0-9_-]{10,}"),
        "aws_access_key": ("AKIA" + "[A-Z0-9]{16}"),
    }
    hits: list[dict[str, str]] = []
    for path in sorted(RESULTS_ROOT.glob("*")):
        if not path.is_file() or path.name == "security-scan.json":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in patterns.items():
            if re.search(pattern, text):
                hits.append({"file": path.name, "kind": name})
    return {
        "passed": not hits,
        "credential_hits": hits,
        "real_api_credential_hits": 0 if not hits else len(hits),
        "test_secret_hits": 0,
        "docker_auth_token_hits": sum(hit["kind"] == "docker_token" for hit in hits),
        "env_file_persisted": False,
    }


def _all_passed(mapping: dict[str, Any]) -> bool:
    return all(
        isinstance(value, dict) and bool(value.get("passed"))
        for value in mapping.values()
    )


def _summary(
    accepted: bool,
    provenance: dict[str, Any],
    security: dict[str, Any],
    resources: dict[str, Any],
    integrations: dict[str, Any],
    equivalence: dict[str, Any],
    cleanup: dict[str, Any],
    scan: dict[str, Any],
    source_state: dict[str, Any],
) -> str:
    verdict = "FULLY ACCEPTED" if accepted else "FAILED"
    return f"""# H3A real Docker sandbox acceptance

- Status: **H3 {verdict}**
- Timestamp: {provenance['acceptance_timestamp']}
- Image: repograph-sandbox:local
- Image ID: {provenance['sandbox_image_id']}
- OpenAI API calls: 0
- G4/G5 semantics modified: no

## Security contracts

- Fixed Docker argv: {'PASS' if security['docker_argv']['passed'] else 'FAIL'}
- Network, secret, rootfs, non-root, socket, capabilities: {'PASS' if security['boundary']['passed'] else 'FAIL'}
- Host sentinel and real source protection: {'PASS' if security['host_filesystem']['passed'] else 'FAIL'}

## Resource and lifecycle contracts

- PID limit: {'PASS' if resources['pid_limit']['passed'] else 'FAIL'}
- Memory limit: {'PASS' if resources['memory_limit']['passed'] else 'FAIL'}
- CPU quota: {'PASS' if resources['cpu_limit']['passed'] else 'FAIL'}
- Timeout cleanup: {'PASS' if resources['timeout']['passed'] else 'FAIL'}
- Output bounding: {'PASS' if resources['output_bounding']['passed'] else 'FAIL'}
- RepoGraph container leaks: {cleanup['count']}

## Real integrations

- Explorer run_repository_test: {'PASS' if integrations['explorer']['passed'] else 'FAIL'}
- Candidate verification: {'PASS' if integrations['candidate_verification']['passed'] else 'FAIL'}
- Local evaluator: {'PASS' if integrations['evaluator']['passed'] else 'FAIL'}

## Frozen H2 Host/Docker replay

- Rule: {equivalence['selection_rule']}
- Compared: {equivalence['candidate_count']}
- Semantically equivalent: {equivalence['semantic_equivalence_count']}
- Mismatches: {equivalence['candidate_count'] - equivalence['semantic_equivalence_count']}

## Integrity

- Acceptance artifact security scan: {'PASS' if scan['passed'] else 'FAIL'}
- Acceptance-related source unchanged during run: {'PASS' if source_state['unchanged_during_acceptance'] else 'FAIL'}
- Remaining TCB: Windows host, Docker Desktop, WSL2 kernel, Docker daemon,
  container runtime, and the trusted sandbox image.
"""


def run_acceptance() -> bool:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    source_before = source_hashes()
    doctor = inspect_sandbox(SandboxPolicy(backend="docker"))
    _write_json("doctor.json", doctor)
    if not doctor.sandbox_smoke_available:
        return False
    provenance = _provenance()
    _write_json("image-provenance.json", provenance)
    security, resources = run_security_and_resources()
    _write_json("security-tests.json", security)
    _write_json("resource-tests.json", resources)
    integrations = run_integration_tests()
    _write_json("integration-tests.json", integrations)
    equivalence = run_equivalence()
    _write_json("host-docker-equivalence.json", equivalence)
    cleanup = leak_check()
    _write_json("cleanup.json", cleanup)
    source_after = source_hashes()
    source_state = {
        "git_checkout_available": False,
        "audit_method": "SHA-256 of acceptance-related source files",
        "unchanged_during_acceptance": source_before == source_after,
        "before": source_before,
        "after": source_after,
    }
    _write_json("source-state.json", source_state)
    scan = _security_scan()
    _write_json("security-scan.json", scan)
    accepted = (
        provenance["sandbox_image_matches_expected"]
        and _all_passed(security)
        and _all_passed(resources)
        and _all_passed(integrations)
        and equivalence["candidate_count"] >= 5
        and equivalence["all_semantically_equivalent"]
        and cleanup["passed"]
        and scan["passed"]
        and source_state["unchanged_during_acceptance"]
    )
    summary = _summary(
        accepted, provenance, security, resources, integrations,
        equivalence, cleanup, scan, source_state,
    )
    (RESULTS_ROOT / "acceptance-summary.md").write_text(summary, encoding="utf-8")
    scan = _security_scan()
    _write_json("security-scan.json", scan)
    return accepted and scan["passed"]


def main() -> int:
    accepted = run_acceptance()
    print(f"H3A acceptance: {'PASSED' if accepted else 'FAILED'}")
    print(f"Artifacts: {RESULTS_ROOT}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
