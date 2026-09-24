"""Deterministic, offline RepoGraph preview demo with real verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal, cast

from engineering_plan import EngineeringPlan, PlannedFileChange, PlannedTest
from observability.models import ReplayObservation
from observability.recorder import TraceRecorder, digest_text, trace_run
from observability.redaction import redact_text
from observability.replay import ReplayBlockedError, replay_artifact
from observability.storage import SQLiteTraceSink
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    PlanExecutionVerification,
    build_candidate_diff,
    materialize_candidate,
    validate_multi_file_candidate,
    verify_candidate_workspace,
)
from repository_exploration import RepositoryExplorationResult
from review_models import CodeReview, OverallRating
from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy
from temporary_workspace import copy_repository_bounded

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = PROJECT_ROOT / "demo" / "fixture_repo"
TASK = "Ignore non-positive cart quantities and add a deterministic summary renderer."
MODEL_NAME = "deterministic/fake-structured-v1"
VERIFICATION_SPEC_DIGEST = digest_text(
    "repograph-demo-v1:ruff+bandit+tests/test_cart.py:docker-no-fallback"
)


def _default_state_root() -> Path:
    configured = os.environ.get("REPOGRAPH_STATE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser() / "demo"
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "RepoGraph" / "demo"
    return Path.home() / ".repograph" / "demo"


def add_demo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sandbox", choices=("docker", "host"), default="docker")
    parser.add_argument("--state-root", type=Path, default=_default_state_root())
    parser.add_argument(
        "--fresh", action="store_true", help="Clear only this demo state namespace."
    )


def _fixture_identity() -> str:
    digest = hashlib.sha256()
    for path in sorted(FIXTURE_ROOT.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(FIXTURE_ROOT).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _plan() -> EngineeringPlan:
    return EngineeringPlan(
        summary="Harden cart totals and add a reusable text summary.",
        files=[
            PlannedFileChange(
                path="repograph_demo/cart.py",
                action="modify",
                rationale="Filter invalid quantities before calculating totals.",
            ),
            PlannedFileChange(
                path="repograph_demo/report.py",
                action="add",
                rationale="Provide the requested stable cart summary format.",
            ),
        ],
        tests=[PlannedTest(
            path="tests/test_cart.py",
            purpose="Verify filtering, totals, and the new summary renderer.",
        )],
        risks=["Filtering policy must be explicit and stable."],
        assumptions=["Non-positive quantities should not contribute to totals."],
    )


def _exploration() -> RepositoryExplorationResult:
    return RepositoryExplorationResult(
        status="completed",
        summary=(
            "Cart totals are implemented in repograph_demo/cart.py; the bounded "
            "test contract is tests/test_cart.py and requires report.py."
        ),
        files_read=["repograph_demo/cart.py", "tests/test_cart.py"],
        searches=["cart_total", "render_summary"],
        search_result_files=["repograph_demo/cart.py", "tests/test_cart.py"],
        tool_call_count=3,
    )


def _candidate() -> MultiFileCandidate:
    return MultiFileCandidate(
        summary="Filter non-positive quantities and add summary rendering.",
        files=[
            CandidateFileChange(
                path="repograph_demo/cart.py",
                action="modify",
                content=(
                    '"""Small cart domain used by the deterministic RepoGraph demo."""\n\n'
                    "from collections.abc import Iterable\n\n\n"
                    "def cart_total(items: Iterable[tuple[str, int, int]]) -> int:\n"
                    "    \"\"\"Return cents for items whose quantities are positive.\"\"\"\n\n"
                    "    return sum(\n"
                    "        price_cents * quantity\n"
                    "        for _, price_cents, quantity in items\n"
                    "        if quantity > 0\n"
                    "    )\n"
                ),
            ),
            CandidateFileChange(
                path="repograph_demo/report.py",
                action="add",
                content=(
                    '"""Deterministic cart reporting."""\n\n'
                    "from collections.abc import Iterable\n\n"
                    "from repograph_demo.cart import cart_total\n\n\n"
                    "def render_summary(items: Iterable[tuple[str, int, int]]) -> str:\n"
                    "    \"\"\"Render a stable one-line total in cents.\"\"\"\n\n"
                    "    materialized = list(items)\n"
                    "    valid_count = sum(\n"
                    "        1 for _, _, quantity in materialized if quantity > 0\n"
                    "    )\n"
                    "    return f\"items={valid_count} total_cents={cart_total(materialized)}\"\n"
                ),
            ),
        ],
    )


def _storage(state_root: Path) -> SQLiteTraceSink:
    root = state_root.expanduser().resolve()
    return SQLiteTraceSink(root / "observability.sqlite3", root / "artifacts")


def _display_state_root(state_root: Path) -> str:
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local:
        try:
            relative = state_root.relative_to(Path(local).resolve())
        except ValueError:
            pass
        else:
            return "$env:LOCALAPPDATA\\" + str(relative)
    return redact_text(str(state_root))


def _observation(verification: PlanExecutionVerification) -> ReplayObservation:
    status = verification.status
    test_result = verification.test_result
    return ReplayObservation(
        status=status,
        resolved=status == "verified",
        exit_classification=(
            f"{test_result.execution_backend}:"
            f"{test_result.status}:"
            f"{test_result.exit_code}"
        ),
    )


def _verify_payload(
    payload: dict[str, object], backend: str
) -> tuple[PlanExecutionVerification, str]:
    plan = EngineeringPlan.model_validate(payload["plan"])
    candidate = MultiFileCandidate.model_validate(payload["candidate"])
    errors = validate_multi_file_candidate(candidate, plan, str(FIXTURE_ROOT))
    if errors:
        raise RuntimeError("Demo candidate validation failed: " + " ".join(errors))
    with tempfile.TemporaryDirectory(prefix="repograph-demo-") as temporary:
        repository = Path(temporary) / "repository"
        copy_repository_bounded(FIXTURE_ROOT, repository)
        materialize_candidate(str(repository), candidate)
        diff = build_candidate_diff(str(FIXTURE_ROOT), str(repository), candidate, plan)
        selected_backend = cast(Literal["host", "docker"], backend)
        verification = verify_candidate_workspace(
            str(repository), candidate, plan,
            sandbox_policy=SandboxPolicy.from_env(selected_backend),
        )
    return verification, diff


def run_demo(args: argparse.Namespace) -> int:
    backend = str(args.sandbox)
    state_root = Path(args.state_root).expanduser().resolve()
    if args.fresh and state_root.exists():
        if state_root.name.casefold() != "demo":
            print("BLOCKED: --fresh may only clear a state directory named 'demo'.")
            return 2
        shutil.rmtree(state_root)
    if backend == "host":
        print("WARNING: Host mode is not a security sandbox.")
    else:
        doctor = inspect_sandbox(SandboxPolicy.from_env("docker"))
        if not doctor.sandbox_smoke_available:
            print("BLOCKED: secure demo requires Docker and the sandbox image.")
            print("Next: python -m sandbox doctor; python -m sandbox build")
            return 2

    plan = _plan()
    exploration = _exploration()
    candidate = _candidate()
    fixture_before = _fixture_identity()
    payload = {
        "schema": "repograph-demo-v1",
        "plan": plan.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
    }
    storage = _storage(state_root)
    recorder = TraceRecorder(storage, required=True)
    run_id = f"demo-{uuid.uuid4()}"
    with trace_run(
        recorder, run_kind="deterministic_demo", repo_identity=fixture_before,
        task=TASK, task_summary=TASK, run_id=run_id,
    ):
        with recorder.span(
            "repository_exploration",
            kind="exploration",
            metadata={"model": MODEL_NAME, "read_only": True},
        ):
            exploration_artifact = recorder.artifact(
                kind="repository_exploration",
                content=exploration.model_dump(mode="json"),
            )
        with recorder.span(
            "structured_planning", kind="planning", metadata={"model": MODEL_NAME}
        ):
            plan_artifact = recorder.artifact(
                kind="engineering_plan", content=plan.model_dump(mode="json")
            )
        with recorder.span(
            "candidate_generation", kind="candidate", metadata={"model": MODEL_NAME}
        ):
            verification, diff = _verify_payload(payload, backend)
            provenance = verification.test_result.sandbox_provenance
            candidate_artifact = recorder.artifact(
                kind="candidate_checkpoint", content=payload,
                metadata={
                    "replay_adapter": "demo_v1",
                    "base_identity": fixture_before,
                    "verification_spec_digest": VERIFICATION_SPEC_DIGEST,
                    "sandbox_backend": backend,
                    "sandbox_image_id": provenance.image_id if provenance else None,
                    "original_result": _observation(verification).model_dump(mode="json"),
                    "apply": False, "git_commit": False, "git_push": False,
                    "create_pr": False,
                },
            )
            diff_artifact = recorder.artifact(kind="candidate_diff", content=diff)
        review = CodeReview(
            overall_rating=(
                OverallRating.GOOD
                if verification.status == "verified"
                else OverallRating.NEEDS_WORK
            ),
            summary=(
                "Deterministic review accepted verified evidence."
                if verification.status == "verified"
                else "Deterministic review requires verification to pass."
            ),
            findings=[],
        )
        verification_artifact = recorder.artifact(
            kind="verification_result", content=verification.model_dump(mode="json")
        )
        review_artifact = recorder.artifact(
            kind="change_set_review", content=review.model_dump(mode="json")
        )
        if not all((exploration_artifact, plan_artifact, candidate_artifact,
                    diff_artifact, verification_artifact, review_artifact)):
            raise RuntimeError("Required demo artifact was not recorded.")
        recorder.link(exploration_artifact, plan_artifact, "derived_from")
        recorder.link(plan_artifact, candidate_artifact, "derived_from")
        recorder.link(candidate_artifact, diff_artifact, "packaged_as")
        recorder.link(candidate_artifact, verification_artifact, "verified_by")
        recorder.link(candidate_artifact, review_artifact, "reviewed_by")

    if _fixture_identity() != fixture_before:
        raise RuntimeError("Demo fixture changed; preview mutation boundary failed.")

    print("RepoGraph deterministic demo")
    print(f"Status: {'PASS' if verification.status == 'verified' else 'FAIL'}")
    print(f"Model: {MODEL_NAME} (offline; 0 LLM calls)")
    print(f"Sandbox: {backend}")
    print("Mutation: preview only; G4/G5 not invoked")
    print("Fixture source unchanged: yes")
    print(f"Plan files: {len(plan.files)}")
    print(f"Exploration: {exploration.status}")
    print(f"Candidate files: {len(candidate.files)}")
    print(f"Verification: {verification.status}")
    print(f"Review: {review.overall_rating.value}")
    print(f"Trace ID: {run_id}")
    print(f"Candidate artifact: {candidate_artifact.artifact_id}")
    display_root = _display_state_root(state_root)
    print(f"State root: {display_root}")
    print("\nInspect:")
    print(f'python -m observability --state-root "{display_root}" show {run_id}')
    print(f'python -m observability --state-root "{display_root}" lineage {run_id}')
    print(f'python -m observability --state-root "{display_root}" export {run_id}')
    print("\nReplay:")
    print(
        f'python -m observability --state-root "{display_root}" replay '
        f"{run_id} --candidate {candidate_artifact.artifact_id}"
    )
    return 0 if verification.status == "verified" else 1


def replay_demo_artifact(
    storage: SQLiteTraceSink, artifact_id: str, *, expected_run_id: str
) -> object:
    record, content = storage.load_artifact(artifact_id)
    if record.run_id != expected_run_id:
        raise ReplayBlockedError("candidate does not belong to the requested run")
    payload = json.loads(content)
    backend = str(record.metadata.get("sandbox_backend", ""))
    if backend not in {"docker", "host"}:
        raise ReplayBlockedError("recorded demo backend is invalid")
    selected_backend = cast(Literal["host", "docker"], backend)
    policy = SandboxPolicy.from_env(selected_backend)
    image_id: str | None = None
    if backend == "docker":
        doctor = inspect_sandbox(policy)
        if not doctor.sandbox_smoke_available:
            raise ReplayBlockedError("recorded Docker environment is unavailable")
        image_id = doctor.sandbox_image_id

    def verify(_content: bytes, _metadata: dict[str, object]) -> ReplayObservation:
        verification, _diff = _verify_payload(payload, backend)
        return _observation(verification)

    return replay_artifact(
        storage, artifact_id, executor=verify,
        current_base_identity=_fixture_identity(), current_backend=backend,
        current_image_id=image_id,
    )
