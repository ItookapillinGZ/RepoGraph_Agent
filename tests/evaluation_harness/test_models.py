"""Evaluation schema tests."""

import unittest

from pydantic import ValidationError

from evaluation.models import EvaluationConfig, EvaluationResult, EvaluationTask


class EvaluationModelTests(unittest.TestCase):
    def test_task_is_strict(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationTask(
                id="x",
                dataset="local",
                repository="repo",
                base_commit="abc",
                task="fix",
                surprise=True,
            )

    def test_task_rejects_empty_or_nul_command(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationTask(
                id="x", dataset="local", repository="r", base_commit="c", task="t", test_command=[]
            )
        with self.assertRaises(ValidationError):
            EvaluationTask(
                id="x", dataset="local", repository="r", base_commit="c", task="t", test_command=["bad\x00arg"]
            )

    def test_config_is_strict_and_validates_dependencies(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationConfig(name="x", unknown=True)
        with self.assertRaises(ValidationError):
            EvaluationConfig(name="x", agentic_explore=False, agentic_test=True)

    def test_disabled_correction_has_zero_effective_budget(self) -> None:
        config = EvaluationConfig(
            name="x", self_correct=False, max_correction_rounds=5
        )
        self.assertEqual(config.effective_correction_rounds, 0)

    def test_config_digest_is_stable_and_changes_with_configuration(self) -> None:
        first = EvaluationConfig(name="x", agentic_test=False)
        same = EvaluationConfig(name="x", agentic_test=False)
        changed = EvaluationConfig(name="x", agentic_test=True)
        self.assertEqual(first.digest(), same.digest())
        self.assertNotEqual(first.digest(), changed.digest())

    def test_legacy_overall_timeout_remains_readable(self) -> None:
        config = EvaluationConfig(
            name="legacy",
            agentic_test=False,
            overall_timeout_seconds=12,
        )
        self.assertEqual(config.effective_task_timeout_seconds, 12)

    def test_result_is_strict_and_missing_telemetry_stays_none(self) -> None:
        result = EvaluationResult(
            task_id="x",
            dataset="local",
            experiment="e",
            repository="r",
            base_commit="c",
            created_at="now",
            status="unresolved",
            duration_seconds=1,
        )
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.estimated_cost_usd)
        with self.assertRaises(ValidationError):
            EvaluationResult.model_validate({**result.model_dump(), "fake": 1})


if __name__ == "__main__":
    unittest.main()
