"""Markdown and JSON report tests."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.models import EvaluationConfig
from evaluation.reports import render_markdown_report, render_summary, write_reports
from evaluation.runner import create_experiment
from tests.evaluation_harness.test_metrics import result


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.experiment = create_experiment(
            "baseline", "local", EvaluationConfig(name="baseline", agentic_test=False)
        )
        self.results = [
            result("one", first=True, final=True),
            result("two", final=True, rounds=1),
        ]

    def test_markdown_has_metadata_metrics_failures_and_per_task_table(self) -> None:
        report = render_markdown_report(self.experiment, self.results)
        self.assertIn("## Experiment metadata", report)
        self.assertIn("## Headline metrics", report)
        self.assertIn("## Execution integrity", report)
        self.assertIn("## LLM usage", report)
        self.assertIn("## External grading", report)
        self.assertIn("`local`: 2 task(s)", report)
        self.assertIn("## Failure breakdown", report)
        self.assertIn("| `one` |", report)
        self.assertIn("no statistical-significance claim", report)

    def test_summary_distinguishes_first_and_final(self) -> None:
        summary = render_summary(self.experiment, self.results)
        self.assertIn("Resolved: 1 / 2", summary)
        self.assertIn("Resolved: 2 / 2", summary)
        self.assertIn("+50.0 percentage points", summary)

    def test_writes_markdown_and_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            markdown, structured = write_reports(
                self.experiment, self.results, Path(temporary)
            )
            self.assertTrue(markdown.is_file())
            payload = json.loads(structured.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["tasks_total"], 2)


if __name__ == "__main__":
    unittest.main()
