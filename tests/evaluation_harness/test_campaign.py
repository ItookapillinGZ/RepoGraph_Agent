"""H2.2 task freezing, budget, and paired-integrity tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.campaign import (
    build_campaign_experiments,
    build_campaign_manifest,
    freeze_local_tasks,
    select_pilot_tasks,
    select_smoke_task,
)
from evaluation.campaign_reports import (
    ComparisonIntegrityError,
    build_paired_comparison,
)
from evaluation.live import LiveBudgetError, LiveRunBudget
from evaluation.models import EvaluationProvenance, EvaluationResult, EvaluationTask
from model_defaults import DEFAULT_PRODUCTION_MODEL


def tasks() -> list[EvaluationTask]:
    categories = [
        "bug_fix",
        "edge_case",
        "small_feature",
        "test_regression",
        "behavioral_refactor",
        "edge_case",
        "bug_fix",
        "multi_file_fix",
        "test_regression",
        "small_feature",
    ]
    return [
        EvaluationTask(
            id=f"task-{index:02d}",
            dataset="local",
            repository=f"repo-{index}",
            base_commit=f"commit-{index}",
            task="fix",
            metadata={"category": category},
        )
        for index, category in enumerate(categories)
    ]


def live_result(task_id: str, config: str, *, resolved: bool) -> EvaluationResult:
    digest = "0" * 64
    return EvaluationResult(
        task_id=task_id,
        dataset="local",
        experiment=config,
        repository="repo",
        base_commit=f"base-{task_id}",
        created_at="2026-09-04T00:00:00+00:00",
        model_name=DEFAULT_PRODUCTION_MODEL,
        configured_model_name=DEFAULT_PRODUCTION_MODEL,
        llm_execution_kind="live",
        status="resolved" if resolved else "unresolved",
        first_attempt_resolved=resolved,
        final_resolved=resolved,
        duration_seconds=2,
        llm_calls=3,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        config_digest=digest,
        provenance=EvaluationProvenance(
            repo_base_commit=f"base-{task_id}",
            config_digest=digest,
            model_name=DEFAULT_PRODUCTION_MODEL,
            configured_model_name=DEFAULT_PRODUCTION_MODEL,
            resolved_model_names=[DEFAULT_PRODUCTION_MODEL],
            model_temperature=0,
            llm_execution_kind="live",
            task_timeout_seconds=900,
            max_correction_rounds=2 if config == "full" else 0,
            timestamp="2026-09-04T00:00:00+00:00",
        ),
    )


class CampaignTests(unittest.TestCase):
    def test_task_set_and_phase_selection_are_deterministic(self) -> None:
        frozen = freeze_local_tasks(list(reversed(tasks())))
        self.assertEqual(
            [item.id for item in frozen], sorted(item.id for item in tasks())
        )
        self.assertEqual(select_smoke_task(frozen).id, "task-07")
        self.assertEqual(
            [item.id for item in select_pilot_tasks(frozen)],
            ["task-00", "task-07", "task-03"],
        )

    def test_manifest_records_same_model_timeout_and_44_runs(self) -> None:
        experiments = build_campaign_experiments("local", task_timeout_seconds=900)
        manifest = build_campaign_manifest(
            tasks(),
            experiments,
            dataset="local",
            task_timeout_seconds=900,
            evaluator_timeout_seconds=300,
            budget_consumed=0,
            repograph_commit=None,
        )
        self.assertEqual(manifest.configured_model, DEFAULT_PRODUCTION_MODEL)
        self.assertEqual(manifest.planned_live_task_runs, 44)
        self.assertEqual(
            {item.config.task_timeout_seconds for item in experiments.values()}, {900}
        )
        self.assertEqual(
            {item.config.model_name for item in experiments.values()},
            {DEFAULT_PRODUCTION_MODEL},
        )

    def test_budget_is_append_only_and_hard_capped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            budget = LiveRunBudget(Path(temporary) / "ledger.jsonl", maximum=2)
            for index in range(2):
                budget.reserve(
                    experiment_id="experiment",
                    task_id=f"task-{index}",
                    config_name="full",
                )
            self.assertEqual(budget.consumed, 2)
            with self.assertRaises(LiveBudgetError):
                budget.reserve(
                    experiment_id="experiment",
                    task_id="task-3",
                    config_name="full",
                )

    def test_paired_matrix_and_deltas_use_same_tasks(self) -> None:
        results = {
            "full": [
                live_result("one", "full", resolved=True),
                live_result("two", "full", resolved=False),
            ],
            "no_correction": [
                live_result("one", "no_correction", resolved=False),
                live_result("two", "no_correction", resolved=False),
            ],
            "no_exploration": [
                live_result("one", "no_exploration", resolved=True),
                live_result("two", "no_exploration", resolved=False),
            ],
            "no_agentic_test": [
                live_result("one", "no_agentic_test", resolved=True),
                live_result("two", "no_agentic_test", resolved=False),
            ],
        }
        comparison = build_paired_comparison(results, resamples=100)
        self.assertTrue(comparison.comparison_valid)
        self.assertEqual(comparison.deltas[0].full_minus_comparison_resolve_rate, 0.5)
        self.assertEqual(comparison.matrix[0].task_id, "one")

    def test_paired_comparison_rejects_task_or_model_mismatch(self) -> None:
        results = {
            key: [live_result("one", key, resolved=True)]
            for key in (
                "full",
                "no_correction",
                "no_exploration",
                "no_agentic_test",
            )
        }
        results["no_agentic_test"][0] = results["no_agentic_test"][0].model_copy(
            update={"configured_model_name": "different"}
        )
        with self.assertRaises(ComparisonIntegrityError):
            build_paired_comparison(results, resamples=10)


if __name__ == "__main__":
    unittest.main()
