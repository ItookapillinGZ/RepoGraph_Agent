"""Offline tests for H2.3 campaign freezing, stops, and artifact security."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.artifact_security import (
    ArtifactSecurityError,
    assert_secret_free_payload,
    scan_persisted_artifacts,
)
from evaluation.campaign import CampaignError, CampaignStopped
from evaluation.campaign_h2_3 import (
    H2_3_FORMAL_TASKS,
    H2_3_PLANNED_BASE_TASK_RUNS,
    H2_3_RETRY_RESERVE,
    H2_3_SMOKE_TASKS,
    _assert_smoke,
    _IntegrityTracker,
    _validate_model_settings,
    build_h2_3_experiments,
    freeze_h2_3_tasks,
)
from evaluation.h2_3_reports import write_h2_3_reports
from evaluation.models import EvaluationResult, EvaluationTask
from evaluation.source_digest import source_tree_digest
from model_defaults import ProductionModelSettings


def model_settings(
    *,
    model: str = "gpt-5.6-sol",
    provider: str = "openai_compatible",
) -> ProductionModelSettings:
    return ProductionModelSettings(
        provider=provider,
        model=model,
        temperature=0.0,
        seed=None,
        api_key=None,
        base_url="https://example.invalid/v1",
        timeout_seconds=60,
        reasoning_effort="none",
        max_completion_tokens=8192,
    )


def result(task_id: str, *, infrastructure: bool) -> EvaluationResult:
    return EvaluationResult(
        task_id=task_id,
        dataset="h2-3",
        experiment="experiment",
        repository="repository",
        base_commit="commit",
        created_at="now",
        status="evaluation_error" if infrastructure else "unresolved",
        final_resolved=False,
        valid_prediction=not infrastructure,
        infrastructure_failure=infrastructure,
        duration_seconds=1,
    )


class H23CampaignContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks = self._tasks()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _tasks(self) -> list[EvaluationTask]:
        values: list[EvaluationTask] = []
        for index in range(20):
            task_id = f"task-{index:02d}"
            repository = self.root / "sources" / task_id
            hidden = self.root / "evaluator-only" / "hidden-tests" / task_id
            repository.mkdir(parents=True)
            hidden.mkdir(parents=True)
            (hidden / "test_hidden.py").write_text(
                "def test_hidden(): assert False\n",
                encoding="utf-8",
            )
            values.append(
                EvaluationTask(
                    id=task_id,
                    dataset="repograph-local-h2-3-v1",
                    repository=str(repository),
                    base_commit=f"{index + 1:040x}",
                    task=f"Fix observable behavior {index}.",
                    test_command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        str(hidden),
                        "-q",
                    ],
                    metadata={
                        "difficulty": ["multi-file", "edge-case"],
                        "requires_multi_file_exploration": True,
                        "repository_file_count": 6,
                    },
                )
            )
        return list(reversed(values))

    def test_freezes_exact_paired_and_disjoint_smoke_sets(self) -> None:
        formal, smoke = freeze_h2_3_tasks(self.tasks)
        self.assertEqual(len(formal), H2_3_FORMAL_TASKS)
        self.assertEqual(len(smoke), H2_3_SMOKE_TASKS)
        self.assertEqual(
            [item.id for item in formal],
            [f"task-{index:02d}" for index in range(17)],
        )
        self.assertTrue(
            {item.id for item in formal}.isdisjoint(item.id for item in smoke)
        )
        self.assertEqual(H2_3_PLANNED_BASE_TASK_RUNS, 70)
        self.assertEqual(H2_3_RETRY_RESERVE, 10)

    def test_all_four_formal_configs_are_live_and_frozen(self) -> None:
        experiments = build_h2_3_experiments(
            "local:test",
            task_timeout_seconds=900,
            evaluator_timeout_seconds=300,
            repograph_commit=None,
            model_settings=model_settings(),
        )
        self.assertEqual(
            set(experiments),
            {
                "smoke",
                "full",
                "no_correction",
                "no_exploration",
                "no_agentic_test",
            },
        )
        for experiment in experiments.values():
            self.assertEqual(experiment.config.llm_execution_kind, "live")
            self.assertEqual(experiment.config.model_name, "gpt-5.6-sol")
            self.assertEqual(experiment.config.task_timeout_seconds, 900)
        self.assertEqual(experiments["no_correction"].config.max_correction_rounds, 0)
        self.assertFalse(experiments["no_exploration"].config.agentic_explore)
        self.assertFalse(experiments["no_agentic_test"].config.agentic_test)

    def test_model_settings_must_match_frozen_protocol(self) -> None:
        _validate_model_settings(model_settings())
        with self.assertRaises(CampaignError):
            _validate_model_settings(model_settings(model="different-model"))


    def test_smoke_accepts_capability_failure_when_exact_grading_is_proven(
        self,
    ) -> None:
        capability_failure = result("capability", infrastructure=False).model_copy(
            update={
                "status": "agent_failed",
                "valid_prediction": False,
                "failure_category": "candidate_generation_failure",
            }
        )
        exact_candidate = result("exact", infrastructure=False).model_copy(
            update={
                "first_attempt_evaluated": True,
                "final_attempt_evaluated": True,
                "candidate_snapshot_paths": ["candidate-01.json"],
            }
        )
        _assert_smoke([capability_failure, exact_candidate])

    def test_smoke_rejects_infrastructure_or_no_exact_candidate(self) -> None:
        capability_failure = result("capability", infrastructure=False).model_copy(
            update={"status": "agent_failed", "valid_prediction": False}
        )
        with self.assertRaises(CampaignStopped):
            _assert_smoke([capability_failure, capability_failure])
        with self.assertRaises(CampaignStopped):
            _assert_smoke(
                [
                    result("infrastructure", infrastructure=True),
                    result("exact", infrastructure=False).model_copy(
                        update={
                            "first_attempt_evaluated": True,
                            "final_attempt_evaluated": True,
                            "candidate_snapshot_paths": ["candidate-01.json"],
                        }
                    ),
                ]
            )
    def test_configuration_stops_at_two_of_first_five_infrastructure_failures(

        self,
    ) -> None:
        tracker = _IntegrityTracker(
            tasks=[],
            source_state={},
            specifications={},
            artifacts_root=self.root / "artifacts",
        )
        observations = [
            result("one", infrastructure=True),
            result("two", infrastructure=False),
            result("three", infrastructure=False),
            result("four", infrastructure=False),
        ]
        for item in observations:
            tracker.observe_final("full", item)
        with self.assertRaises(CampaignStopped):
            tracker.observe_final("full", result("five", infrastructure=True))

    def test_partial_infrastructure_results_render_truthful_reports(self) -> None:
        formal, _smoke = freeze_h2_3_tasks(self.tasks)
        results_by_config: dict[str, list[EvaluationResult]] = {}
        for name in (
            "full",
            "no_correction",
            "no_exploration",
            "no_agentic_test",
        ):
            first = result(formal[0].id, infrastructure=False).model_copy(
                update={"experiment": name}
            )
            second = result(
                formal[1].id,
                infrastructure=name == "no_exploration",
            ).model_copy(update={"experiment": name})
            results_by_config[name] = [first, second]
        paths = write_h2_3_reports(
            self.root / "reports",
            formal,
            results_by_config,
            artifacts_root=self.root / "artifacts",
            regrade_workspace_root=self.root / "regrade",
        )
        self.assertEqual(len(paths), 4)
        comparison = (self.root / "reports" / "comparison.json").read_text(
            encoding="utf-8"
        )
        self.assertIn('"not_comparable_pairs": 1', comparison)
        self.assertIn('"infrastructure_failures": 1', comparison)


class ArtifactSecurityTests(unittest.TestCase):
    def test_configured_secret_is_rejected_without_echoing_value(self) -> None:
        secret = "test-secret-material-that-must-not-appear"
        with self.assertRaises(ArtifactSecurityError) as raised:
            assert_secret_free_payload(
                f"payload={secret}",
                {"OPENAI_API_KEY": secret},
            )
        self.assertNotIn(secret, str(raised.exception))

    def test_key_like_material_is_rejected(self) -> None:
        with self.assertRaises(ArtifactSecurityError):
            assert_secret_free_payload("token sk-abcdefghijklmnop1234")

    def test_safe_artifact_scan_reports_only_counts_and_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result.json").write_text('{"status": "ok"}', encoding="utf-8")
            (root / "binary.bin").write_bytes(b"ignored")
            report = scan_persisted_artifacts(
                [root],
                {"OPENAI_API_KEY": "configured-secret-value"},
            )
        self.assertEqual(report.files_scanned, 1)
        self.assertFalse(report.secret_detected)
        self.assertEqual(len(report.rules), 2)


class SourceDigestTests(unittest.TestCase):
    def test_campaign_results_are_excluded_but_source_changes_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            campaign = root / "evaluation" / "campaigns" / "result.json"
            campaign.parent.mkdir(parents=True)
            campaign.write_text('{"run": 1}\n', encoding="utf-8")
            initial = source_tree_digest(root)
            campaign.write_text('{"run": 2}\n', encoding="utf-8")
            self.assertEqual(source_tree_digest(root), initial)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(source_tree_digest(root), initial)


if __name__ == "__main__":
    unittest.main()
