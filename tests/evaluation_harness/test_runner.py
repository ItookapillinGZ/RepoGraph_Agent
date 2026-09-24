"""Evaluation runner, isolation, resume, and timeout tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engineering_plan import EngineeringPlan, PlannedFileChange
from evaluation.models import EvaluationConfig, EvaluationResult, EvaluationWorkerResult
from evaluation.runner import create_experiment, run_evaluation_task, run_experiment
from evaluation.storage import EvaluationStorage
from evaluation.workspace import (
    apply_candidate_exact,
    extract_candidate_patch,
    prepare_task_workspace,
)
from plan_execution import CandidateFileChange, MultiFileCandidate
from tests.evaluation_harness.helpers import (
    FIXED,
    WRONG,
    StaticAdapter,
    candidate,
    git,
    make_repository,
    repository_digest,
    task,
)


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.commit = make_repository(self.root)
        self.workspace_root = self.root / "evaluation-workspaces"
        self.config = EvaluationConfig(name="baseline", agentic_test=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_task(self, adapter: StaticAdapter, **config_updates: object):
        config = self.config.model_copy(update=config_updates)
        return run_evaluation_task(
            task(self.source, self.commit),
            config,
            workspace_root=str(self.workspace_root),
            experiment_id="experiment",
            adapter=adapter,
        )

    def test_external_evaluator_determines_resolved(self) -> None:
        result = self.run_task(
            StaticAdapter(candidate(FIXED), review_good=False, verification_good=False)
        )
        self.assertEqual(result.status, "resolved")
        self.assertTrue(result.final_resolved)
        self.assertFalse(result.review_good)

    def test_repograph_good_does_not_make_wrong_candidate_resolved(self) -> None:
        result = self.run_task(StaticAdapter(candidate(WRONG), review_good=True))
        self.assertEqual(result.status, "unresolved")
        self.assertFalse(result.final_resolved)
        self.assertTrue(result.review_good)

    def test_first_and_final_candidates_are_evaluated_without_second_llm_call(self) -> None:
        adapter = StaticAdapter(
            candidate(FIXED), first=candidate(WRONG), rounds=1
        )
        result = self.run_task(adapter)
        self.assertFalse(result.first_attempt_resolved)
        self.assertTrue(result.final_resolved)
        self.assertEqual(result.correction_rounds_used, 1)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(result.test_calls, 3)

    def test_no_correction_reuses_one_external_evaluation(self) -> None:
        result = self.run_task(StaticAdapter(candidate(FIXED)))
        self.assertTrue(result.first_attempt_resolved)
        self.assertTrue(result.final_resolved)
        self.assertEqual(result.test_calls, 2)

    def test_invalid_base_commit_is_infrastructure_error(self) -> None:
        evaluation_task = task(self.source, "deadbeef")
        result = run_evaluation_task(
            evaluation_task,
            self.config,
            workspace_root=str(self.workspace_root),
            adapter=StaticAdapter(candidate()),
        )
        self.assertEqual(result.status, "evaluation_error")
        self.assertEqual(result.failure_category, "infrastructure_error")

    def test_source_repository_is_unchanged(self) -> None:
        before = repository_digest(self.source)
        head = git(self.source, "rev-parse", "HEAD")
        status = git(self.source, "status", "--porcelain=v1")
        self.run_task(StaticAdapter(candidate(FIXED)))
        self.assertEqual(repository_digest(self.source), before)
        self.assertEqual(git(self.source, "rev-parse", "HEAD"), head)
        self.assertEqual(git(self.source, "status", "--porcelain=v1"), status)

    def test_workspace_is_fresh_for_same_task_retry(self) -> None:
        workspace, _ = prepare_task_workspace(
            str(self.source), self.commit, self.workspace_root, "e", "task"
        )
        (workspace / "pollution.tmp").write_text("polluted", encoding="utf-8")
        fresh, _ = prepare_task_workspace(
            str(self.source), self.commit, self.workspace_root, "e", "task"
        )
        self.assertEqual(fresh, workspace)
        self.assertFalse((fresh / "pollution.tmp").exists())

    def test_distinct_tasks_have_distinct_workspaces(self) -> None:
        first, _ = prepare_task_workspace(
            str(self.source), self.commit, self.workspace_root, "e", "one"
        )
        second, _ = prepare_task_workspace(
            str(self.source), self.commit, self.workspace_root, "e", "two"
        )
        self.assertNotEqual(first, second)

    def test_workspace_inside_source_is_rejected_before_mutation(self) -> None:
        before = repository_digest(self.source)
        result = run_evaluation_task(
            task(self.source, self.commit),
            self.config,
            workspace_root=str(self.source / "unsafe-evaluation"),
            adapter=StaticAdapter(candidate(FIXED)),
        )
        self.assertEqual(result.status, "evaluation_error")
        self.assertFalse((self.source / "unsafe-evaluation").exists())
        self.assertEqual(repository_digest(self.source), before)

    def test_patch_is_git_generated_and_contains_only_candidate_paths(self) -> None:
        result = self.run_task(StaticAdapter(candidate(FIXED)))
        self.assertEqual(result.changed_files, ["src/calculator.py"])
        self.assertIn("diff --git a/src/calculator.py b/src/calculator.py", result.model_patch or "")
        self.assertTrue((result.model_patch or "").endswith("\n"))
        self.assertNotIn("evaluation.db", result.model_patch or "")
        self.assertNotIn(".env", result.model_patch or "")
        self.assertEqual(result.config_digest, self.config.digest())
        self.assertIsNotNone(result.prediction_sha256)
        self.assertEqual(result.provenance and result.provenance.repo_base_commit, self.commit)

    def test_patch_includes_candidate_add_and_delete_without_untracked_noise(self) -> None:
        workspace, resolved = prepare_task_workspace(
            str(self.source), self.commit, self.workspace_root, "e", "add-delete"
        )
        exact_plan = EngineeringPlan(
            summary="Replace the calculator module.",
            files=[
                PlannedFileChange(
                    path="src/calculator.py", action="delete", rationale="Remove old API."
                ),
                PlannedFileChange(
                    path="src/replacement.py", action="add", rationale="Add replacement API."
                ),
            ],
        )
        exact_candidate = MultiFileCandidate(
            summary="Replace the module.",
            files=[
                CandidateFileChange(
                    path="src/calculator.py", action="delete", content=None
                ),
                CandidateFileChange(
                    path="src/replacement.py",
                    action="add",
                    content="def add(a, b):\n    return a + b\n",
                ),
            ],
        )
        apply_candidate_exact(workspace, exact_plan, exact_candidate)
        (workspace / "untracked.tmp").write_text("noise", encoding="utf-8")
        patch, changed = extract_candidate_patch(workspace, resolved, exact_candidate)
        self.assertEqual(changed, ["src/calculator.py", "src/replacement.py"])
        self.assertIn("deleted file mode", patch)
        self.assertIn("new file mode", patch)
        self.assertNotIn("untracked.tmp", patch)

    @patch("evaluation.runner.run_evaluation_worker")
    def test_hard_task_timeout_is_recorded(self, worker) -> None:
        worker.return_value = EvaluationWorkerResult(
            status="timeout",
            failure_reason="worker tree terminated",
        )
        result = run_evaluation_task(
            task(self.source, self.commit),
            self.config.model_copy(update={"task_timeout_seconds": 0.05}),
            workspace_root=str(self.workspace_root),
            experiment_id="experiment",
        )
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.failure_category, "timeout")
        self.assertTrue(result.process_isolated)
        self.assertTrue(result.hard_timeout)

    def test_evaluator_timeout_is_separate(self) -> None:
        evaluation_task = task(self.source, self.commit).model_copy(
            update={
                "test_command": [sys.executable, "-c", "import time; time.sleep(0.2)"]
            }
        )
        config = self.config.model_copy(update={"evaluator_timeout_seconds": 0.05})
        result = run_evaluation_task(
            evaluation_task,
            config,
            workspace_root=str(self.workspace_root),
            adapter=StaticAdapter(candidate(FIXED)),
        )
        self.assertEqual(result.status, "timeout")
        self.assertIn("Evaluator exceeded", result.failure_reason or "")

    def test_missing_evaluator_command_is_evaluation_error(self) -> None:
        evaluation_task = task(self.source, self.commit).model_copy(
            update={"test_command": None}
        )
        result = run_evaluation_task(
            evaluation_task,
            self.config,
            workspace_root=str(self.workspace_root),
            adapter=StaticAdapter(candidate(FIXED)),
        )
        self.assertEqual(result.status, "evaluation_error")

    def test_pytest_usage_error_is_not_counted_as_agent_test_failure(self) -> None:
        evaluation_task = task(self.source, self.commit).model_copy(
            update={
                "test_command": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/does_not_exist.py",
                    "-q",
                ]
            }
        )
        result = run_evaluation_task(
            evaluation_task,
            self.config,
            workspace_root=str(self.workspace_root),
            adapter=StaticAdapter(candidate(FIXED)),
        )
        self.assertEqual(result.status, "evaluation_error")
        self.assertEqual(result.failure_category, "external_evaluator_failure")
        self.assertEqual(result.evaluator_failure_kind, "usage_error")

    def test_task_failure_does_not_abort_experiment(self) -> None:
        storage = EvaluationStorage(self.root / "db.sqlite", self.root / "results")
        experiment = create_experiment("run", "local", self.config)
        bad = task(self.source, "bad-commit", task_id="bad")
        good = task(self.source, self.commit, task_id="good")
        results = run_experiment(
            [bad, good],
            experiment,
            workspace_root=str(self.workspace_root),
            storage=storage,
            adapter=StaticAdapter(candidate(FIXED)),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual({item.status for item in results}, {"evaluation_error", "resolved"})

    @patch("evaluation.runner.run_evaluation_worker")
    def test_process_timeout_is_persisted_and_next_task_runs(self, worker) -> None:
        storage = EvaluationStorage(self.root / "isolated.sqlite", self.root / "isolated-results")
        experiment = create_experiment("isolated", "local", self.config)
        first = task(self.source, self.commit, task_id="first")
        second = task(self.source, self.commit, task_id="second")

        def outcome(request, **_kwargs):
            if request.task.id == "first":
                return EvaluationWorkerResult(status="timeout", failure_reason="terminated")
            return EvaluationWorkerResult(
                status="completed",
                result=EvaluationResult(
                    task_id="second",
                    dataset=second.dataset,
                    experiment=experiment.id,
                    repository=second.repository,
                    base_commit=second.base_commit,
                    created_at="now",
                    status="resolved",
                    duration_seconds=1,
                    process_isolated=True,
                ),
            )

        worker.side_effect = outcome
        results = run_experiment(
            [first, second],
            experiment,
            workspace_root=str(self.workspace_root),
            storage=storage,
        )
        by_id = {item.task_id: item for item in results}
        self.assertTrue(by_id["first"].hard_timeout)
        self.assertEqual(by_id["second"].status, "resolved")
        self.assertEqual(storage.get_result(experiment.id, "first").status, "timeout")

    @patch("evaluation.runner.run_evaluation_worker")
    def test_resume_after_process_timeout_retries_only_timed_out_task(self, worker) -> None:
        storage = EvaluationStorage(self.root / "resume.sqlite", self.root / "resume-results")
        experiment = create_experiment("resume", "local", self.config)
        evaluation_task = task(self.source, self.commit)
        worker.return_value = EvaluationWorkerResult(status="timeout", failure_reason="terminated")
        run_experiment(
            [evaluation_task],
            experiment,
            workspace_root=str(self.workspace_root),
            storage=storage,
        )
        worker.return_value = EvaluationWorkerResult(
            status="completed",
            result=EvaluationResult(
                task_id=evaluation_task.id,
                dataset=evaluation_task.dataset,
                experiment=experiment.id,
                repository=evaluation_task.repository,
                base_commit=evaluation_task.base_commit,
                created_at="now",
                status="resolved",
                duration_seconds=1,
                process_isolated=True,
            ),
        )
        run_experiment(
            [evaluation_task],
            experiment,
            workspace_root=str(self.workspace_root),
            storage=storage,
            rerun_failed=True,
        )
        self.assertEqual(storage.get_result(experiment.id, evaluation_task.id).status, "resolved")

    def test_resume_skips_completed_and_rerun_failed_retries_only_failures(self) -> None:
        storage = EvaluationStorage(self.root / "db.sqlite", self.root / "results")
        experiment = create_experiment("run", "local", self.config)
        evaluation_task = task(self.source, self.commit)
        failing = StaticAdapter(None)
        run_experiment(
            [evaluation_task], experiment, workspace_root=str(self.workspace_root),
            storage=storage, adapter=failing
        )
        self.assertEqual(failing.calls, 1)
        skipped = StaticAdapter(candidate(FIXED))
        run_experiment(
            [evaluation_task], experiment, workspace_root=str(self.workspace_root),
            storage=storage, adapter=skipped
        )
        self.assertEqual(skipped.calls, 0)
        run_experiment(
            [evaluation_task], experiment, workspace_root=str(self.workspace_root),
            storage=storage, adapter=skipped, rerun_failed=True
        )
        self.assertEqual(skipped.calls, 1)
        self.assertEqual(storage.list_results(experiment.id)[0].status, "resolved")

    def test_test_command_is_fixed_argv_not_shell(self) -> None:
        marker = self.root / "should-not-exist"
        evaluation_task = task(self.source, self.commit).model_copy(
            update={"test_command": [sys.executable, "-c", f"import sys; sys.exit(1); {marker}"]}
        )
        result = run_evaluation_task(
            evaluation_task, self.config, workspace_root=str(self.workspace_root),
            adapter=StaticAdapter(candidate(FIXED))
        )
        self.assertEqual(result.status, "unresolved")
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
