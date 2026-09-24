import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from static_analysis import (
    STATIC_ANALYSIS_TIMEOUT_SECONDS,
    ToolCategory,
    ToolSeverity,
    analyze_code,
)


def completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def empty_bandit_output() -> str:
    return json.dumps({"errors": [], "results": []})


class StaticAnalysisTests(unittest.TestCase):
    @patch("static_analysis.subprocess.run")
    def test_ruff_json_is_normalized_and_exit_one_is_not_failure(
        self,
        mocked_run,
    ):
        ruff_output = json.dumps(
            [
                {
                    "code": "F821",
                    "message": "Undefined name `missing_name`",
                    "location": {"row": 2, "column": 5},
                }
            ]
        )
        mocked_run.side_effect = [
            completed(ruff_output, returncode=1),
            completed(empty_bandit_output()),
        ]

        result = analyze_code("print(missing_name)", "python")

        self.assertEqual(result.tools_run, ["ruff", "bandit"])
        self.assertEqual(result.tool_errors, [])
        finding = result.findings[0]
        self.assertEqual(finding.tool, "ruff")
        self.assertEqual(finding.rule_id, "F821")
        self.assertEqual(finding.category, ToolCategory.BUG)
        self.assertEqual(finding.severity, ToolSeverity.HIGH)
        self.assertEqual(finding.line_number, 2)
        self.assertEqual(finding.column, 5)

        ruff_call = mocked_run.call_args_list[0]
        self.assertEqual(
            ruff_call.args[0][:6],
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format",
                "json",
            ],
        )
        self.assertIn("--no-cache", ruff_call.args[0])
        self.assertEqual(
            ruff_call.args[0][ruff_call.args[0].index("--target-version") + 1],
            "py39",
        )
        self.assertFalse(ruff_call.kwargs["shell"])
        self.assertEqual(
            ruff_call.kwargs["timeout"],
            STATIC_ANALYSIS_TIMEOUT_SECONDS,
        )
        temp_path = Path(ruff_call.args[0][-1])
        self.assertEqual(ruff_call.kwargs["cwd"], str(temp_path.parent))
        self.assertFalse(temp_path.exists())

    @patch("static_analysis.subprocess.run")
    def test_bandit_json_is_normalized_and_exit_one_is_not_failure(
        self,
        mocked_run,
    ):
        bandit_output = json.dumps(
            {
                "errors": [],
                "results": [
                    {
                        "test_id": "B602",
                        "issue_severity": "HIGH",
                        "issue_confidence": "HIGH",
                        "issue_text": "subprocess call with shell=True",
                        "line_number": 14,
                        "col_offset": 0,
                    }
                ],
            }
        )
        mocked_run.side_effect = [
            completed("[]"),
            completed(bandit_output, returncode=1),
        ]

        result = analyze_code("subprocess.run(cmd, shell=True)", "python")

        self.assertEqual(result.tools_run, ["ruff", "bandit"])
        self.assertEqual(result.tool_errors, [])
        finding = result.findings[0]
        self.assertEqual(finding.tool, "bandit")
        self.assertEqual(finding.rule_id, "B602")
        self.assertEqual(finding.category, ToolCategory.SECURITY)
        self.assertEqual(finding.severity, ToolSeverity.HIGH)
        self.assertEqual(finding.confidence, "high")
        self.assertEqual(finding.line_number, 14)
        self.assertEqual(finding.column, 1)

        bandit_call = mocked_run.call_args_list[1]
        self.assertEqual(
            bandit_call.args[0][:5],
            [sys.executable, "-m", "bandit", "-f", "json"],
        )
        self.assertFalse(bandit_call.kwargs["shell"])

    @patch.dict("os.environ", {"PATH": ""})
    def test_tools_run_through_current_interpreter_without_path(self):
        result = analyze_code("answer = 42\n", "python")

        self.assertEqual(result.tools_run, ["ruff", "bandit"])
        self.assertEqual(result.tool_errors, [])

    @patch("static_analysis.subprocess.run")
    def test_missing_tools_are_errors_not_zero_findings(self, mocked_run):
        mocked_run.side_effect = [FileNotFoundError(), FileNotFoundError()]

        result = analyze_code("return 42", "python")

        self.assertEqual(result.findings, [])
        self.assertEqual(result.tools_run, [])
        self.assertEqual(
            result.tool_errors,
            ["ruff executable not found", "bandit executable not found"],
        )

    @patch("static_analysis.subprocess.run")
    def test_timeout_is_recorded_without_stopping_other_tool(self, mocked_run):
        mocked_run.side_effect = [
            subprocess.TimeoutExpired(
                cmd=["ruff"],
                timeout=STATIC_ANALYSIS_TIMEOUT_SECONDS,
            ),
            completed(empty_bandit_output()),
        ]

        result = analyze_code("return 42", "python")

        self.assertEqual(result.tools_run, ["bandit"])
        self.assertEqual(len(result.tool_errors), 1)
        self.assertIn("ruff timed out", result.tool_errors[0])

    @patch("static_analysis.subprocess.run")
    def test_malformed_json_is_recorded_for_each_tool(self, mocked_run):
        mocked_run.side_effect = [
            completed("not-json"),
            completed("{"),
        ]

        result = analyze_code("return 42", "python")

        self.assertEqual(result.tools_run, [])
        self.assertEqual(len(result.tool_errors), 2)
        self.assertIn("ruff returned malformed JSON", result.tool_errors[0])
        self.assertIn("bandit returned malformed JSON", result.tool_errors[1])

    @patch("static_analysis.subprocess.run")
    def test_real_invocation_error_is_not_treated_as_findings(self, mocked_run):
        mocked_run.side_effect = [
            completed("[]", returncode=2, stderr="invalid option"),
            completed(empty_bandit_output()),
        ]

        result = analyze_code("return 42", "python")

        self.assertEqual(result.tools_run, ["bandit"])
        self.assertEqual(result.findings, [])
        self.assertEqual(
            result.tool_errors,
            ["ruff failed with exit code 2: invalid option"],
        )

    @patch("static_analysis.subprocess.run")
    def test_non_python_language_does_not_run_tools(self, mocked_run):
        result = analyze_code("const answer = 42;", "javascript")

        mocked_run.assert_not_called()
        self.assertEqual(result.findings, [])
        self.assertEqual(result.tools_run, [])
        self.assertEqual(result.tool_errors, [])


if __name__ == "__main__":
    unittest.main()
