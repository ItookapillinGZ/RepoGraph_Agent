"""Deterministic Docker integration and frozen H2 replay for H3A."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from engineering_plan import EngineeringPlan, PlannedFileChange, PlannedTest
from evaluation.candidate_artifacts import candidate_sha256
from evaluation.evaluator import run_local_evaluator
from evaluation.models import EvaluationTask
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    materialize_candidate,
    verify_candidate_workspace,
)
from repository_exploration import explore_repository
from sandbox.acceptance_security import hash_tree, run_command
from sandbox.policy import SandboxPolicy
from temporary_workspace import copy_repository_bounded

PROJECT_ROOT = Path(__file__).resolve().parent.parent
H2_RESULTS = (
    PROJECT_ROOT / "evaluation" / "campaigns"
    / "h2_3e_openai_luna_recovery" / "results.jsonl"
)
H2_DATASET = (
    PROJECT_ROOT / ".evaluation" / "h2_3_benchmark" / "h2-3-local-v1.jsonl"
)
H2_ARTIFACTS = (
    PROJECT_ROOT / ".evaluation" / "h2_3d_openai_luna_runtime" / "artifacts"
)
H2_HIDDEN = (
    PROJECT_ROOT / ".evaluation" / "h2_3_benchmark"
    / "evaluator-only" / "hidden-tests"
)
EXPECTED_IMAGE_ID = (
    "sha256:c66fc8697a59ed31747845eb432dd05ae3371d12a4ed23286708cc8d67f96d57"
)
SELECTION_RULE = (
    "From the authoritative H2.3E recovery results, use config=no_exploration; "
    "sort unique task IDs alphabetically; select the first four, then include "
    "registry-alias-collision. Selection is frozen before Docker replay and is "
    "independent of stored or replayed pass/fail outcomes."
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_integration_tests() -> dict[str, Any]:
    policy = SandboxPolicy(backend="docker", max_timeout_seconds=30)
    integrations: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="repograph-h3a-explorer-") as directory:
        root = Path(directory)
        (root / "tests").mkdir()
        (root / "tests" / "test_container.py").write_text(
            "import os\n"
            "def test_inside_docker():\n"
            "    assert os.path.exists('/.dockerenv')\n"
            "    assert os.getuid() == 10001\n",
            encoding="utf-8",
        )
        model = MagicMock()
        bound = model.bind_tools.return_value
        bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "run_repository_test",
                    "args": {"test_file": "tests/test_container.py"},
                    "id": "h3a-explorer-test",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Docker test evidence collected."),
        ]
        result = explore_repository(
            str(root), "Run the single authorized deterministic test.",
            allowed_test_files=["tests/test_container.py"], model=model,
            max_tool_calls=2, max_test_tool_calls=1, sandbox_policy=policy,
        )
        integrations["explorer"] = {
            "passed": result.status == "completed"
            and result.tests_run == ["tests/test_container.py"]
            and result.test_tool_call_count == 1
            and "Status: PASSED" in result.summary,
            "status": result.status,
            "tests_run": result.tests_run,
            "test_tool_call_count": result.test_tool_call_count,
            "deterministic_model": True,
            "container_assertion": "/.dockerenv and uid=10001",
        }

    with tempfile.TemporaryDirectory(prefix="repograph-h3a-verify-") as directory:
        root = Path(directory)
        (root / "tests").mkdir()
        (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "tests" / "test_app.py").write_text(
            "import os\n"
            "def test_candidate_inside_docker():\n"
            "    assert os.path.exists('/.dockerenv')\n"
            "    assert os.getuid() == 10001\n",
            encoding="utf-8",
        )
        candidate = MultiFileCandidate(
            summary="Deterministic Docker verification fixture.",
            files=[CandidateFileChange(
                path="app.py", action="modify", content="VALUE = 1\n"
            )],
        )
        plan = EngineeringPlan(
            summary="Verify the deterministic candidate.",
            files=[PlannedFileChange(
                path="app.py", action="modify", rationale="Acceptance fixture."
            )],
            tests=[PlannedTest(
                path="tests/test_app.py", purpose="Prove Docker execution."
            )],
        )
        result = verify_candidate_workspace(
            str(root), candidate, plan, sandbox_policy=policy
        )
        provenance = result.test_result.sandbox_provenance
        integrations["candidate_verification"] = {
            "passed": result.status == "verified"
            and result.test_result.status == "passed"
            and result.test_result.execution_backend == "docker"
            and result.test_result.sandboxed
            and provenance is not None
            and provenance.image_id == EXPECTED_IMAGE_ID,
            "status": result.status,
            "test_status": result.test_result.status,
            "backend": result.test_result.execution_backend,
            "sandboxed": result.test_result.sandboxed,
            "provenance": _jsonable(provenance) if provenance else None,
        }

    with tempfile.TemporaryDirectory(prefix="repograph-h3a-evaluator-") as directory:
        root = Path(directory)
        (root / "tests").mkdir()
        (root / "tests" / "test_eval.py").write_text(
            "import os\n"
            "def test_evaluator_inside_docker():\n"
            "    assert os.path.exists('/.dockerenv')\n"
            "    assert os.getuid() == 10001\n",
            encoding="utf-8",
        )
        task = EvaluationTask(
            id="h3a-docker-evaluator", dataset="h3a-acceptance",
            repository=str(root), base_commit="deterministic-fixture",
            task="Prove the evaluator runs inside Docker.",
            test_command=[sys.executable, "-m", "pytest", "tests/test_eval.py", "-q"],
        )
        outcome = run_local_evaluator(
            task, root, timeout_seconds=30, max_output_chars=2_000,
            sandbox_policy=policy,
        )
        provenance = outcome.sandbox_provenance
        integrations["evaluator"] = {
            "passed": outcome.status == "passed"
            and provenance is not None
            and provenance.backend == "docker"
            and provenance.sandboxed
            and provenance.image_id == EXPECTED_IMAGE_ID,
            "status": outcome.status,
            "failure_kind": outcome.failure_kind,
            "provenance": _jsonable(provenance) if provenance else None,
        }
    return integrations


def _selected_candidates() -> list[dict[str, Any]]:
    records = [
        row for row in _load_jsonl(H2_RESULTS) if row["config"] == "no_exploration"
    ]
    by_task = {row["result"]["task_id"]: row for row in records}
    ordered = sorted(by_task)
    selected_ids = ordered[:4]
    if "registry-alias-collision" not in selected_ids:
        selected_ids.append("registry-alias-collision")
    if len(selected_ids) != 5:
        raise RuntimeError("Frozen H2 selection did not produce exactly five tasks.")
    return [by_task[task_id] for task_id in selected_ids]


def _prepare_workspace(
    task: EvaluationTask,
    candidate: MultiFileCandidate,
    destination: Path,
) -> tuple[str, str]:
    source = Path(task.repository).resolve(strict=True)
    git_head = run_command(["git", "-C", str(source), "rev-parse", "HEAD"])
    if not git_head["ok"]:
        raise RuntimeError(f"Could not verify fixture base for {task.id}.")
    actual_head = git_head["stdout"].strip()
    if actual_head != task.base_commit:
        raise RuntimeError(
            f"Fixture base mismatch for {task.id}: {actual_head} != {task.base_commit}"
        )
    copy_repository_bounded(source, destination)
    materialize_candidate(str(destination), candidate)
    hidden_source = H2_HIDDEN / task.id
    hidden_digest = hash_tree(hidden_source)
    shutil.copytree(hidden_source, destination / "repograph_hidden_tests")
    return actual_head, hidden_digest


def _replay(
    task: EvaluationTask,
    candidate: MultiFileCandidate,
    backend: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"repograph-h3a-{backend}-") as directory:
        workspace = Path(directory) / "workspace"
        base_commit, hidden_digest = _prepare_workspace(
            task, candidate, workspace
        )
        replay_task = task.model_copy(update={
            "repository": str(workspace),
            "test_command": [
                sys.executable, "-m", "pytest", "tests",
                "repograph_hidden_tests", "-q",
            ],
        })
        outcome = run_local_evaluator(
            replay_task, workspace, timeout_seconds=120, max_output_chars=5_000,
            sandbox_policy=SandboxPolicy(
                backend=backend, max_timeout_seconds=120,
                max_output_chars=5_000,
            ),
        )
    return {
        "resolved": outcome.status == "passed",
        "status": outcome.status,
        "failure_category": outcome.failure_category,
        "failure_kind": outcome.failure_kind,
        "exit_code": outcome.exit_code,
        "base_commit": base_commit,
        "hidden_evaluator_sha256": hidden_digest,
        "provenance": (
            _jsonable(outcome.sandbox_provenance)
            if outcome.sandbox_provenance else None
        ),
    }


def run_equivalence() -> dict[str, Any]:
    tasks = {
        row["id"]: EvaluationTask.model_validate(row) for row in _load_jsonl(H2_DATASET)
    }
    selections = _selected_candidates()
    frozen = [{
        "task_id": row["result"]["task_id"],
        "config": row["config"],
        "candidate_snapshot": row["result"]["candidate_snapshot_paths"][-1],
        "correction_rounds_used": row["result"]["correction_rounds_used"],
    } for row in selections]
    comparisons: list[dict[str, Any]] = []
    for selected in frozen:
        task_id = selected["task_id"]
        snapshot_path = H2_ARTIFACTS / selected["candidate_snapshot"]
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        candidate = MultiFileCandidate.model_validate(snapshot["candidate"])
        actual_digest = candidate_sha256(candidate)
        if actual_digest != snapshot["candidate_sha256"]:
            raise RuntimeError(f"Candidate digest mismatch for {task_id}.")
        host = _replay(tasks[task_id], candidate, "host")
        docker = _replay(tasks[task_id], candidate, "docker")
        comparisons.append({
            **selected,
            "candidate_sha256": actual_digest,
            "host": host,
            "docker": docker,
            "semantically_equivalent": host["resolved"] == docker["resolved"],
            "classification_equivalent": (
                host["status"], host["failure_category"], host["failure_kind"]
            ) == (
                docker["status"], docker["failure_category"], docker["failure_kind"]
            ),
        })
    equivalent = sum(item["semantically_equivalent"] for item in comparisons)
    return {
        "selection_rule": SELECTION_RULE,
        "selection_frozen_before_docker_replay": True,
        "selected": frozen,
        "comparisons": comparisons,
        "candidate_count": len(comparisons),
        "semantic_equivalence_count": equivalent,
        "all_semantically_equivalent": equivalent == len(comparisons),
    }
