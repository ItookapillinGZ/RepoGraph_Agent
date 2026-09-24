"""Strict H2.2 stopped-campaign resume tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.campaign import (
    _write_json,
    build_campaign_experiments,
    build_campaign_manifest,
    freeze_local_tasks,
)
from evaluation.campaign_resume import (
    CAMPAIGN_PHASES,
    _phase_tasks,
    resume_live_campaign,
)
from evaluation.live import LivePreflightResult, LiveRunBudget
from evaluation.storage import EvaluationStorage
from model_defaults import get_production_model_settings
from tests.evaluation_harness.test_campaign import live_result, tasks


class CampaignResumeTests(unittest.TestCase):
    def test_resume_reuses_preflight_and_runs_only_missing_tasks(self) -> None:
        settings = get_production_model_settings({})
        frozen = freeze_local_tasks(tasks())
        dataset = "local"
        experiments = build_campaign_experiments(
            dataset,
            task_timeout_seconds=900,
            evaluator_timeout_seconds=300,
            model_settings=settings,
        )
        manifest = build_campaign_manifest(
            frozen,
            experiments,
            dataset=dataset,
            task_timeout_seconds=900,
            evaluator_timeout_seconds=300,
            budget_consumed=0,
            repograph_commit=None,
            model_settings=settings,
        )
        manifest.status = "stopped"
        manifest.stop_reason = "transient provider outage"
        preflight = LivePreflightResult(
            structured_output_succeeded=True,
            configured_model=settings.model,
            resolved_models=[settings.model],
            model_temperature=settings.temperature,
            model_seed=settings.seed,
            configured_provider=settings.provider,
            api_base_url=settings.base_url,
            reasoning_effort=settings.reasoning_effort,
            max_completion_tokens=settings.max_completion_tokens,
            llm_calls=1,
            telemetry_incomplete=False,
            timestamp="2026-09-06T00:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            output = root / "output"
            storage = EvaluationStorage(
                workspace / "evaluation.db", workspace / "results"
            )
            budget = LiveRunBudget(workspace / "live-task-runs.jsonl", maximum=45)
            _write_json(output / "manifest.json", manifest)
            _write_json(output / "preflight.json", preflight)
            for phase, experiment in experiments.items():
                if phase != "no_agentic_test":
                    storage.save_experiment(experiment)

            planned = [
                (phase, task)
                for phase in CAMPAIGN_PHASES
                for task in _phase_tasks(frozen)[phase]
            ]
            for phase, task in planned:
                if phase == "no_agentic_test":
                    continue
                experiment = experiments[phase]
                storage.save_task(task)
                budget.reserve(
                    experiment_id=experiment.id,
                    task_id=task.id,
                    config_name=experiment.config.name,
                )
                result = live_result(task.id, phase, resolved=False).model_copy(
                    update={
                        "experiment": experiment.id,
                        "repository": task.repository,
                        "base_commit": task.base_commit,
                        "process_isolated": True,
                    }
                )
                storage.save_result(result)

            def run_missing(
                task,
                _config,
                *,
                workspace_root,
                artifacts_root,
                experiment_id,
                adapter,
                git_commit,
                execution_attempt,
            ):
                del workspace_root, artifacts_root, adapter, git_commit
                return live_result(
                    task.id, "no_agentic_test", resolved=False
                ).model_copy(
                    update={
                        "experiment": experiment_id,
                        "repository": task.repository,
                        "base_commit": task.base_commit,
                        "process_isolated": True,
                        "execution_attempt": execution_attempt,
                    }
                )

            with (
                patch("evaluation.campaign.run_live_preflight") as live_preflight,
                patch("evaluation.campaign_resume.source_snapshots", return_value={}),
                patch("evaluation.campaign.assert_source_snapshots"),
                patch(
                    "evaluation.campaign_resume.write_campaign_reports",
                    return_value=[],
                ),
                patch(
                    "evaluation.runner.run_evaluation_task",
                    side_effect=run_missing,
                ) as task_run,
            ):
                outcome = resume_live_campaign(
                    frozen,
                    dataset=dataset,
                    workspace_root=workspace,
                    output_root=output,
                    maximum_live_task_runs=45,
                    task_timeout_seconds=900,
                    evaluator_timeout_seconds=300,
                    model_settings=settings,
                )

            live_preflight.assert_not_called()
            self.assertEqual(task_run.call_count, 10)
            self.assertEqual(outcome.live_task_runs_consumed, 44)
            self.assertEqual(budget.consumed, 44)
            completed = type(manifest).model_validate_json(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed.status, "complete")
            self.assertIsNone(completed.stop_reason)


if __name__ == "__main__":
    unittest.main()
