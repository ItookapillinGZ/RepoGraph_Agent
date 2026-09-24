import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from fix_verification import (
    FixVerificationResult,
    WorkspaceCopyError,
    copy_repository_bounded,
    verify_candidate_fix,
)
from repository_context import RepoContext, RepositoryFile
from static_analysis import (
    StaticAnalysisResult,
    ToolCategory,
    ToolFinding,
    ToolSeverity,
)
from test_execution import TestRunResult


class TemporaryFixRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.repository = self.base / "repo"
        self.repository.mkdir()
        self.original_code = "def add(a, b):\n    return a - b\n"
        self.updated_code = "def add(a, b):\n    return a + b\n"
        self.write_file("src/calculator.py", self.original_code)
        self.write_file(
            "tests/test_calculator.py",
            "from src.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        self.context = RepoContext(
            repository_root=str(self.repository.resolve()),
            target_file="src/calculator.py",
            related_files=[
                RepositoryFile(
                    path="tests/test_calculator.py",
                    relationship="test",
                    content="def test_add(): pass\n",
                )
            ],
        )
        self.patch_text = (
            "--- a/src/calculator.py\n+++ b/src/calculator.py\n"
            "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n"
            "+    return a + b\n"
        )

    def write_file(self, relative_path: str, content: str = "data") -> Path:
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def verify(
        self,
        test_result: TestRunResult,
        static_result: StaticAnalysisResult | None = None,
        **options,
    ) -> FixVerificationResult:
        with (
            patch(
                "fix_verification.analyze_code",
                return_value=static_result or StaticAnalysisResult(),
            ),
            patch(
                "fix_verification.execute_targeted_tests",
                return_value=test_result,
            ),
        ):
            return verify_candidate_fix(
                str(self.repository),
                self.context,
                "src/calculator.py",
                self.updated_code,
                self.patch_text,
                **options,
            )


class WorkspaceCopyTests(TemporaryFixRepository):
    def test_regular_repository_files_are_copied(self) -> None:
        destination = self.base / "copy"
        result = copy_repository_bounded(self.repository, destination)
        self.assertEqual(
            (destination / "src/calculator.py").read_text(encoding="utf-8"),
            self.original_code,
        )
        self.assertEqual(result.files_copied, 2)

    def test_excluded_directories_are_not_copied(self) -> None:
        for directory in (".git", ".venv", "__pycache__", "node_modules", "dist"):
            self.write_file(f"{directory}/ignored.txt")
        destination = self.base / "copy"
        copy_repository_bounded(self.repository, destination)
        for directory in (".git", ".venv", "__pycache__", "node_modules", "dist"):
            self.assertFalse((destination / directory).exists())

    def test_secret_and_environment_files_are_not_copied(self) -> None:
        for file_name in (".env", ".env.local", "server.pem", "private.KEY"):
            self.write_file(file_name, "secret")
        destination = self.base / "copy"
        result = copy_repository_bounded(self.repository, destination)
        for file_name in (".env", ".env.local", "server.pem", "private.KEY"):
            self.assertFalse((destination / file_name).exists())
        self.assertTrue(any("secret/environment" in item for item in result.warnings))

    def test_file_budget_aborts_complete_copy(self) -> None:
        with self.assertRaisesRegex(WorkspaceCopyError, "file budget"):
            copy_repository_bounded(
                self.repository,
                self.base / "copy",
                max_files=1,
            )

    def test_byte_budget_aborts_complete_copy(self) -> None:
        with self.assertRaisesRegex(WorkspaceCopyError, "byte budget"):
            copy_repository_bounded(
                self.repository,
                self.base / "copy",
                max_bytes=1,
            )

    def test_destination_inside_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkspaceCopyError, "outside the source"):
            copy_repository_bounded(self.repository, self.repository / "copy")

    def test_directory_symlink_is_skipped_when_supported(self) -> None:
        external = self.base / "external"
        external.mkdir()
        (external / "outside.py").write_text("SECRET = 1\n", encoding="utf-8")
        link = self.repository / "linked"
        try:
            os.symlink(external, link, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Directory symlinks are unavailable: {error}")
        destination = self.base / "copy"
        result = copy_repository_bounded(self.repository, destination)
        self.assertFalse((destination / "linked").exists())
        self.assertTrue(any("symlinks" in item for item in result.warnings))

    def test_file_symlink_is_skipped_when_supported(self) -> None:
        external = self.base / "outside.py"
        external.write_text("SECRET = 1\n", encoding="utf-8")
        link = self.repository / "linked.py"
        try:
            os.symlink(external, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"File symlinks are unavailable: {error}")
        destination = self.base / "copy"
        result = copy_repository_bounded(self.repository, destination)
        self.assertFalse((destination / "linked.py").exists())
        self.assertTrue(any("symlinks" in item for item in result.warnings))


class CandidateVerificationTests(TemporaryFixRepository):
    def test_schema_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            FixVerificationResult.model_validate(
                {"status": "verified", "unexpected": True}
            )

    def test_passing_tests_and_clean_static_analysis_are_verified(self) -> None:
        result = self.verify(TestRunResult(status="passed", framework="pytest"))
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.patch, self.patch_text)

    def test_candidate_is_written_only_in_temporary_repository(self) -> None:
        original_bytes = (self.repository / "src/calculator.py").read_bytes()

        def inspect_temporary_root(root, _context, _language, *, enabled):
            self.assertTrue(enabled)
            self.assertNotEqual(Path(root).resolve(), self.repository.resolve())
            self.assertEqual(
                (Path(root) / "src/calculator.py").read_text(encoding="utf-8"),
                self.updated_code,
            )
            return TestRunResult(status="passed", framework="pytest")

        with (
            patch(
                "fix_verification.analyze_code",
                return_value=StaticAnalysisResult(),
            ) as analyzer,
            patch(
                "fix_verification.execute_targeted_tests",
                side_effect=inspect_temporary_root,
            ) as executor,
        ):
            result = verify_candidate_fix(
                str(self.repository),
                self.context,
                "src/calculator.py",
                self.updated_code,
                self.patch_text,
            )

        self.assertEqual(result.status, "verified")
        analyzer.assert_called_once_with(self.updated_code, "python")
        executor.assert_called_once()
        self.assertEqual(
            (self.repository / "src/calculator.py").read_bytes(),
            original_bytes,
        )

    def test_failing_tests_mark_candidate_failed(self) -> None:
        result = self.verify(TestRunResult(status="failed", framework="pytest"))
        self.assertEqual(result.status, "failed")

    def test_timed_out_tests_are_not_verified(self) -> None:
        result = self.verify(TestRunResult(status="timed_out", framework="pytest"))
        self.assertEqual(result.status, "not_verified")

    def test_test_infrastructure_error_is_not_candidate_failure(self) -> None:
        result = self.verify(TestRunResult(status="error", framework="pytest"))
        self.assertEqual(result.status, "not_verified")

    def test_no_related_tests_are_not_verified(self) -> None:
        result = self.verify(TestRunResult(status="not_run"))
        self.assertEqual(result.status, "not_verified")

    def test_high_static_bug_blocks_verified_status(self) -> None:
        analysis = StaticAnalysisResult(
            findings=[
                ToolFinding(
                    tool="ruff",
                    rule_id="F821",
                    category=ToolCategory.BUG,
                    severity=ToolSeverity.HIGH,
                    message="Undefined name.",
                    line_number=1,
                )
            ],
            tools_run=["ruff", "bandit"],
        )
        result = self.verify(
            TestRunResult(status="passed", framework="pytest"),
            analysis,
        )
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("ruff:F821" in item for item in result.warnings))

    def test_high_static_security_finding_blocks_verified_status(self) -> None:
        analysis = StaticAnalysisResult(
            findings=[
                ToolFinding(
                    tool="bandit",
                    rule_id="B602",
                    category=ToolCategory.SECURITY,
                    severity=ToolSeverity.HIGH,
                    message="Unsafe shell call.",
                    line_number=1,
                )
            ],
            tools_run=["ruff", "bandit"],
        )
        result = self.verify(
            TestRunResult(status="passed", framework="pytest"),
            analysis,
        )
        self.assertEqual(result.status, "failed")

    def test_low_style_finding_does_not_block_verified_status(self) -> None:
        analysis = StaticAnalysisResult(
            findings=[
                ToolFinding(
                    tool="ruff",
                    rule_id="E501",
                    category=ToolCategory.STYLE,
                    severity=ToolSeverity.LOW,
                    message="Line too long.",
                    line_number=1,
                )
            ],
            tools_run=["ruff", "bandit"],
        )
        result = self.verify(
            TestRunResult(status="passed", framework="pytest"),
            analysis,
        )
        self.assertEqual(result.status, "verified")

    def test_static_tool_error_prevents_verified_claim(self) -> None:
        result = self.verify(
            TestRunResult(status="passed", framework="pytest"),
            StaticAnalysisResult(tool_errors=["ruff executable not found"]),
        )
        self.assertEqual(result.status, "not_verified")

    def test_target_parent_traversal_is_rejected_before_copy(self) -> None:
        result = verify_candidate_fix(
            str(self.repository),
            self.context,
            "../outside.py",
            self.updated_code,
            self.patch_text,
        )
        self.assertEqual(result.status, "error")
        self.assertTrue(any("parent traversal" in item for item in result.warnings))

    def test_invalid_candidate_syntax_is_rejected_before_tools(self) -> None:
        with (
            patch("fix_verification.analyze_code") as analyzer,
            patch("fix_verification.execute_targeted_tests") as executor,
        ):
            result = verify_candidate_fix(
                str(self.repository),
                self.context,
                "src/calculator.py",
                "def broken(:\n",
                self.patch_text,
            )
        self.assertEqual(result.status, "error")
        analyzer.assert_not_called()
        executor.assert_not_called()

    def test_workspace_budget_error_is_structured(self) -> None:
        result = self.verify(
            TestRunResult(status="passed", framework="pytest"),
            max_workspace_bytes=1,
        )
        self.assertEqual(result.status, "error")
        self.assertTrue(any("byte budget" in item for item in result.warnings))


@unittest.skipUnless(
    importlib.util.find_spec("pytest") is not None,
    "pytest is not installed in the current interpreter",
)
class RealCandidateVerificationTests(TemporaryFixRepository):
    def test_real_pytest_passes_only_for_temporary_candidate(self) -> None:
        original_bytes = (self.repository / "src/calculator.py").read_bytes()
        with patch(
            "fix_verification.analyze_code",
            return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
        ):
            result = verify_candidate_fix(
                str(self.repository),
                self.context,
                "src/calculator.py",
                self.updated_code,
                self.patch_text,
            )
        self.assertEqual(
            result.status,
            "verified",
            result.test_result.stdout + result.test_result.stderr,
        )
        self.assertEqual(
            (self.repository / "src/calculator.py").read_bytes(),
            original_bytes,
        )


if __name__ == "__main__":
    unittest.main()
