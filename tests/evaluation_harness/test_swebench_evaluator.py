"""Official SWE-bench adapter tests without Docker image builds."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.adapters.swebench_evaluator import (
    SWEBenchPrerequisites,
    evaluate_swebench_prediction,
    prediction_sha256,
    swebench_run_id,
)
from evaluation.process_runner import FixedProcessOutcome

READY = SWEBenchPrerequisites(
    swebench_installed=True,
    swebench_version="4.1.0",
    docker_cli_available=True,
    docker_daemon_available=True,
    docker_version="29.4.3",
)


class SWEBenchEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predictions = self.root / "predictions.jsonl"
        self.write_prediction("patch A")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_prediction(self, patch_text: str, **updates: object) -> None:
        record = {
            "instance_id": "django__django-1",
            "model_name_or_path": "repograph/test",
            "model_patch": patch_text,
            **updates,
        }
        self.predictions.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def evaluate(self):
        return evaluate_swebench_prediction(
            self.predictions,
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            instance_id="django__django-1",
            experiment_id="baseline",
            timeout_seconds=30,
            output_root=self.root / "official",
        )

    @staticmethod
    def process(*, timeout: bool = False, returncode: int = 0) -> FixedProcessOutcome:
        return FixedProcessOutcome(
            returncode=returncode,
            timed_out=timeout,
            stdout="",
            stderr="failure" if returncode else "",
            output_truncated=False,
            duration_seconds=1,
        )

    def report_runner(self, *, resolved: bool, wrong_instance: bool = False):
        def run(argv, **kwargs):
            run_id = argv[argv.index("--run_id") + 1]
            instance_id = "other__repo-1" if wrong_instance else argv[argv.index("--instance_ids") + 1]
            report = {
                "schema_version": 2,
                "completed_ids": [instance_id],
                "resolved_ids": [instance_id] if resolved else [],
                "unresolved_ids": [] if resolved else [instance_id],
                "error_ids": [],
            }
            path = Path(kwargs["cwd"]) / f"repograph__test.{run_id}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            return self.process()

        return run

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites")
    def test_optional_dependency_missing_is_evaluation_error(self, detect) -> None:
        detect.return_value = READY.model_copy(update={"swebench_installed": False})
        result = self.evaluate()
        self.assertEqual(result.status, "evaluation_error")
        self.assertIn("not installed", result.failure_reason or "")

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites")
    def test_docker_unavailable_is_evaluation_error(self, detect) -> None:
        detect.return_value = READY.model_copy(update={"docker_daemon_available": False})
        result = self.evaluate()
        self.assertEqual(result.status, "evaluation_error")
        self.assertIn("Docker", result.failure_reason or "")

    def test_invalid_prediction_is_rejected_before_harness(self) -> None:
        self.write_prediction("")
        result = self.evaluate()
        self.assertEqual(result.status, "evaluation_error")
        self.assertIn("Invalid prediction", result.failure_reason or "")

    def test_patch_digest_and_run_identity_are_deterministic_and_patch_bound(self) -> None:
        self.assertEqual(prediction_sha256("a"), prediction_sha256("a"))
        first = swebench_run_id("e", "i", "patch A")
        self.assertEqual(first, swebench_run_id("e", "i", "patch A"))
        self.assertNotEqual(first, swebench_run_id("e", "i", "patch B"))

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites", return_value=READY)
    @patch("evaluation.adapters.swebench_evaluator.run_fixed_argv")
    def test_official_command_is_fixed_argv_and_resolved_report_wins(
        self, run, _detect
    ) -> None:
        run.side_effect = self.report_runner(resolved=True)
        result = self.evaluate()
        self.assertEqual(result.status, "resolved")
        self.assertTrue(result.completed)
        self.assertTrue(result.resolved)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], [__import__("sys").executable, "-m", "swebench.harness.run_evaluation"])
        self.assertEqual(argv[argv.index("--max_workers") + 1], "1")
        self.assertNotIn("shell", run.call_args.kwargs)

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites", return_value=READY)
    @patch("evaluation.adapters.swebench_evaluator.run_fixed_argv")
    def test_completed_unresolved_report(self, run, _detect) -> None:
        run.side_effect = self.report_runner(resolved=False)
        result = self.evaluate()
        self.assertEqual(result.status, "unresolved")
        self.assertFalse(result.resolved)

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites", return_value=READY)
    @patch("evaluation.adapters.swebench_evaluator.run_fixed_argv")
    def test_evaluator_timeout(self, run, _detect) -> None:
        run.return_value = self.process(timeout=True)
        self.assertEqual(self.evaluate().status, "timeout")

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites", return_value=READY)
    @patch("evaluation.adapters.swebench_evaluator.run_fixed_argv")
    def test_harness_nonzero_is_infrastructure_failure(self, run, _detect) -> None:
        run.return_value = self.process(returncode=2)
        result = self.evaluate()
        self.assertEqual(result.status, "evaluation_error")
        self.assertIn("nonzero", result.failure_reason or "")

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites", return_value=READY)
    @patch("evaluation.adapters.swebench_evaluator.run_fixed_argv")
    def test_malformed_or_wrong_instance_report_is_rejected(self, run, _detect) -> None:
        def malformed(argv, **kwargs):
            run_id = argv[argv.index("--run_id") + 1]
            (Path(kwargs["cwd"]) / f"repograph__test.{run_id}.json").write_text(
                "{}", encoding="utf-8"
            )
            return self.process()

        run.side_effect = malformed
        self.assertEqual(self.evaluate().status, "evaluation_error")
        run.side_effect = self.report_runner(resolved=True, wrong_instance=True)
        self.assertEqual(self.evaluate().status, "evaluation_error")

    @patch("evaluation.adapters.swebench_evaluator.detect_swebench_prerequisites", return_value=READY)
    @patch("evaluation.adapters.swebench_evaluator.run_fixed_argv")
    def test_cache_metadata_digest_mismatch_is_rejected(self, run, _detect) -> None:
        run_id = swebench_run_id("baseline", "django__django-1", "patch A")
        metadata_root = self.root / "official" / "repograph-swebench-metadata"
        metadata_root.mkdir(parents=True)
        (metadata_root / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "dataset_name": "princeton-nlp/SWE-bench_Lite",
                    "split": "test",
                    "instance_id": "django__django-1",
                    "prediction_sha256": prediction_sha256("different"),
                    "run_id": run_id,
                    "swebench_version": "4.1.0",
                    "created_at": "now",
                }
            ),
            encoding="utf-8",
        )
        result = self.evaluate()
        self.assertEqual(result.status, "evaluation_error")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
