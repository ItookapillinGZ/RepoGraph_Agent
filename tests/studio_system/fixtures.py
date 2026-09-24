"""Deterministic models and real temporary Git repositories for Studio tests."""

from __future__ import annotations

import atexit
import hashlib
import os
import subprocess  # nosec B404 - test-only fixed argv Git harness
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from change_set import ChangeSetReview
from engineering_plan import (
    EngineeringPlan,
    PlannedFileChange,
    PlannedTest,
    build_engineering_plan_graph,
)
from git_delivery import validate_local_git_delivery
from github_delivery import GitHubRemoteDeliveryResult
from plan_application import PlanApplicationBundle
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    build_plan_execution_graph,
)
from review_models import OverallRating

_boot_directory = tempfile.TemporaryDirectory(prefix="repograph-h11-import-")
atexit.register(_boot_directory.cleanup)
_boot_root = Path(_boot_directory.name)
_boot_workspace = _boot_root / "workspace"
_boot_data = _boot_root / "data"
_boot_workspace.mkdir()
_boot_data.mkdir()
os.environ.setdefault("REPOGRAPH_WORKSPACE_ROOT", str(_boot_workspace))
os.environ.setdefault("REPOGRAPH_STUDIO_DATA_DIR", str(_boot_data))
os.environ.setdefault(
    "REPOGRAPH_STUDIO_ALLOWED_ORIGIN",
    "http://127.0.0.1:3000",
)
_project_venv_scripts = Path(__file__).resolve().parents[2] / ".venv" / "Scripts"
if _project_venv_scripts.is_dir():
    os.environ["PATH"] = (
        f"{_project_venv_scripts}{os.pathsep}{os.environ.get('PATH', '')}"
    )

from studio_backend.app import StudioServices, create_app
from studio_backend.approvals import StudioApprovalService
from studio_backend.config import StudioConfig
from studio_backend.execution import StudioExecutionAdapter

TASK = "Fix the add function so the existing regression test passes."
ORIGINAL_SOURCE = (
    "def add(a: int, b: int) -> int:\n"
    "    return a - b\n"
)
FIXED_SOURCE = (
    "def add(a: int, b: int) -> int:\n"
    "    return a + b\n"
)
BUGGY_SOURCE = (
    "def add(a: int, b: int) -> int:\n"
    "    return a * b\n"
)
TEST_SOURCE = (
    "from src.calculator import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)
GITHUB_REMOTE_URL = "https://github.com/repograph-tests/calculator.git"
SYSTEM_TEST_TIMEOUT_SECONDS = 120


def run_git(repository: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - fixed test harness argv
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def initialize_repository(
    workspace: Path,
    *,
    with_bare_remote: bool = False,
) -> tuple[Path, Path | None, str]:
    repository = workspace / "calculator"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "src" / "calculator.py").write_bytes(
        ORIGINAL_SOURCE.encode("utf-8")
    )
    (repository / "tests" / "test_calculator.py").write_bytes(
        TEST_SOURCE.encode("utf-8")
    )
    run_git(repository, "init", "-b", "main")
    run_git(repository, "config", "core.autocrlf", "false")
    run_git(repository, "config", "user.name", "RepoGraph System Test")
    run_git(repository, "config", "user.email", "repograph@example.invalid")
    run_git(repository, "add", "src/calculator.py", "tests/test_calculator.py")
    run_git(repository, "commit", "-m", "AAA")
    initial_sha = run_git(repository, "rev-parse", "HEAD")

    bare_remote: Path | None = None
    if with_bare_remote:
        bare_remote = workspace.parent / "remote.git"
        bare_remote.mkdir()
        run_git(bare_remote, "init", "--bare")
        run_git(repository, "push", bare_remote.resolve().as_uri(), "main")
        run_git(repository, "remote", "add", "origin", GITHUB_REMOTE_URL)
    return repository, bare_remote, initial_sha


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_call(name: str, arguments: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def engineering_plan() -> EngineeringPlan:
    return EngineeringPlan(
        summary="Fix the calculator regression without changing its API.",
        files=[
            PlannedFileChange(
                path="src/calculator.py",
                action="modify",
                rationale="Use addition as required by the regression test.",
            )
        ],
        tests=[
            PlannedTest(
                path="tests/test_calculator.py",
                purpose="Verify that add(2, 3) returns 5.",
            )
        ],
        risks=[],
        assumptions=[],
    )


def candidate(source: str = FIXED_SOURCE) -> MultiFileCandidate:
    return MultiFileCandidate(
        summary="Corrected calculator addition.",
        files=[
            CandidateFileChange(
                path="src/calculator.py",
                action="modify",
                content=source,
            )
        ],
    )


def _explorer_model() -> MagicMock:
    model = MagicMock()
    model.bind_tools.return_value.invoke.side_effect = [
        tool_call(
            "read_repository_file",
            {"path": "src/calculator.py"},
            "read-source",
        ),
        tool_call(
            "read_repository_file",
            {"path": "tests/test_calculator.py"},
            "read-test",
        ),
        AIMessage(content="Calculator implementation and regression test located."),
    ]
    return model


def _structured_model(*responses: object) -> MagicMock:
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = list(responses)
    return model


def _reviewer(*ratings: OverallRating) -> Callable[..., ChangeSetReview]:
    remaining = list(ratings or (OverallRating.GOOD,))

    def review(*_args: object, **_kwargs: object) -> ChangeSetReview:
        rating = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return ChangeSetReview(
            overall_rating=rating,
            summary=(
                "Candidate requires correction."
                if rating != OverallRating.GOOD
                else "Candidate matches the approved plan."
            ),
            file_results=[],
        )

    return review


def deterministic_services(
    *,
    correction: bool = False,
    remote_boundary: Callable[..., GitHubRemoteDeliveryResult] | None = None,
) -> StudioServices:
    planner = _structured_model(engineering_plan())
    planning_graph = build_engineering_plan_graph(
        explorer_model=_explorer_model(),
        planner_model=planner,
    )
    if correction:
        executor = _structured_model(candidate(BUGGY_SOURCE))
        corrector = _structured_model(candidate(FIXED_SOURCE))
        reviewer = _reviewer(OverallRating.NEEDS_WORK, OverallRating.GOOD)
    else:
        executor = _structured_model(candidate(FIXED_SOURCE))
        corrector = _structured_model(candidate(FIXED_SOURCE))
        reviewer = _reviewer(OverallRating.GOOD)
    execution_graph = build_plan_execution_graph(
        executor_model=executor,
        corrector_model=corrector,
        change_set_reviewer=reviewer,
    )

    def execution_adapter_factory(storage, repositories):
        return StudioExecutionAdapter(
            storage,
            repositories,
            planning_graph=planning_graph,
            execution_graph=execution_graph,
        )

    if remote_boundary is None:
        return StudioServices(execution_adapter_factory=execution_adapter_factory)

    def approval_service_factory(storage, repositories):
        return StudioApprovalService(
            storage,
            repositories,
            remote_boundary=remote_boundary,
        )

    return StudioServices(
        execution_adapter_factory=execution_adapter_factory,
        approval_service_factory=approval_service_factory,
    )


def build_test_app(
    workspace: Path,
    data_dir: Path,
    *,
    correction: bool = False,
    remote_boundary: Callable[..., GitHubRemoteDeliveryResult] | None = None,
) -> FastAPI:
    data_dir.mkdir(parents=True, exist_ok=True)
    config = StudioConfig(
        workspace_root=workspace.resolve(),
        data_dir=data_dir.resolve(),
        allowed_origin="http://127.0.0.1:3000",
        max_concurrent_runs=1,
    )
    return create_app(
        config,
        services=deterministic_services(
            correction=correction,
            remote_boundary=remote_boundary,
        ),
    )


def start_and_wait_verified(
    client: TestClient,
    *,
    repository_id: str = "calculator",
    correction: bool = False,
) -> dict[str, object]:
    response = client.post(
        "/api/runs",
        json={
            "repository_id": repository_id,
            "task": TASK,
            "self_correct": correction,
            "max_correction_rounds": 1 if correction else 0,
        },
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    deadline = time.monotonic() + SYSTEM_TEST_TIMEOUT_SECONDS
    detail: dict[str, object] = {}
    while time.monotonic() < deadline:
        detail_response = client.get(f"/api/runs/{run_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()
        if detail["status"] in {
            "verified",
            "failed",
            "error",
            "stale",
            "conflict",
        }:
            break
        time.sleep(0.05)
    if detail.get("status") != "verified":
        execution = client.get(
            f"/api/runs/{run_id}/artifacts/plan_execution_result"
        )
        events = client.get(f"/api/runs/{run_id}/events")
        detail["debug_execution"] = (
            execution.json() if execution.status_code == 200 else execution.text
        )
        detail["debug_events"] = (
            events.json() if events.status_code == 200 else events.text
        )
    assert detail.get("status") == "verified", detail
    while time.monotonic() < deadline:
        event_response = client.get(f"/api/runs/{run_id}/events")
        assert event_response.status_code == 200, event_response.text
        events = event_response.json()["events"]
        if events and events[-1]["event_type"] == "waiting_for_apply_approval":
            break
        time.sleep(0.01)
    else:
        raise AssertionError(
            "verified run did not persist its final apply-approval event before timeout"
        )
    return detail


def assert_event_order(events: list[dict[str, object]], expected: list[str]) -> None:
    cursor = -1
    for event_type in expected:
        for index in range(cursor + 1, len(events)):
            if events[index].get("event_type") == event_type:
                cursor = index
                break
        else:
            raise AssertionError(
                f"Missing or out-of-order event {event_type!r}: "
                f"{[event.get('event_type') for event in events]}"
            )


class InjectedRemoteBoundary:
    """Recording G5B adapter for Studio wiring and recovery tests."""

    def __init__(
        self,
        *,
        push_created: bool = True,
        pr_created: bool = True,
    ) -> None:
        self.push_created = push_created
        self.pr_created = pr_created
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        repository_root: str,
        bundle: PlanApplicationBundle,
        *,
        approved: bool = False,
        local_branch: str | None = None,
        remote_name: str = "origin",
        base_branch: str,
        pr_title: str | None = None,
        pr_body: str | None = None,
    ) -> GitHubRemoteDeliveryResult:
        delivery = validate_local_git_delivery(
            repository_root,
            bundle,
            branch_name=local_branch,
        )
        self.calls.append(
            {
                "repository_root": repository_root,
                "approved": approved,
                "local_branch": local_branch,
                "remote_name": remote_name,
                "base_branch": base_branch,
                "pr_title": pr_title,
                "pr_body": pr_body,
            }
        )
        return GitHubRemoteDeliveryResult(
            status="published",
            remote_name=remote_name,
            owner="repograph-tests",
            repository="calculator",
            branch_name=delivery.branch_name,
            commit_sha=delivery.commit_sha,
            base_sha=delivery.base_sha,
            base_branch=base_branch,
            approval_digest=delivery.approval_digest,
            pushed=True,
            push_created=self.push_created,
            pr_created=self.pr_created,
            pr_number=42,
            pr_url="https://github.com/repograph-tests/calculator/pull/42",
        )
