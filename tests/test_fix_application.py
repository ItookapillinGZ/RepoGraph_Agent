import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from fix_application import (
    ApplyFixResult,
    apply_verified_fix,
    content_fingerprint,
)
from fix_context import CodeFix, build_fix_patch
from fix_verification import FixVerificationResult, verify_candidate_fix
from repository_context import RepoContext, RepositoryFile
from static_analysis import StaticAnalysisResult
from test_execution import TestRunResult


class TemporaryApplicationRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.repository = self.base / "repo"
        self.target = self.repository / "src/calculator.py"
        self.target.parent.mkdir(parents=True)
        self.original_code = "def add(a, b):\n    return a - b\n"
        self.updated_code = "def add(a, b):\n    return a + b\n"
        self.target.write_bytes(self.original_code.encode("utf-8"))
        self.original_hash = content_fingerprint(self.target.read_bytes())
        self.fix = CodeFix(
            summary="Correct addition.",
            addressed_findings=["Wrong arithmetic operation"],
            updated_code=self.updated_code,
        )
        self.verification = FixVerificationResult(
            status="verified",
            static_analysis=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
            test_result=TestRunResult(status="passed", framework="pytest"),
            patch=build_fix_patch(
                self.original_code,
                self.updated_code,
                "src/calculator.py",
            ),
        )

    def apply(self, **changes) -> ApplyFixResult:
        arguments = {
            "repository_root": str(self.repository),
            "target_file": "src/calculator.py",
            "original_code": self.original_code,
            "original_content_hash": self.original_hash,
            "fix": self.fix,
            "verification": self.verification,
            "fix_validation_errors": [],
            "enabled": True,
            "auto_fix": True,
        }
        arguments.update(changes)
        return apply_verified_fix(**arguments)


class ApplyFixSchemaAndPreconditionTests(TemporaryApplicationRepository):
    def test_result_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ApplyFixResult.model_validate(
                {"status": "applied", "unexpected": True}
            )

    def test_apply_disabled_never_writes(self) -> None:
        before = self.target.read_bytes()
        result = self.apply(enabled=False)
        self.assertEqual(result.status, "not_requested")
        self.assertEqual(self.target.read_bytes(), before)

    def test_apply_requires_auto_fix_mode(self) -> None:
        before = self.target.read_bytes()
        result = self.apply(auto_fix=False)
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), before)

    def test_unverified_candidate_never_writes(self) -> None:
        before = self.target.read_bytes()
        result = self.apply(
            verification=FixVerificationResult(status="not_verified")
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), before)

    def test_failed_candidate_never_writes(self) -> None:
        before = self.target.read_bytes()
        result = self.apply(verification=FixVerificationResult(status="failed"))
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), before)

    def test_missing_candidate_never_writes(self) -> None:
        before = self.target.read_bytes()
        result = self.apply(fix=None)
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), before)

    def test_validation_errors_prevent_write(self) -> None:
        before = self.target.read_bytes()
        result = self.apply(fix_validation_errors=["candidate invalid"])
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), before)

    def test_unchanged_candidate_prevents_write(self) -> None:
        unchanged = self.fix.model_copy(update={"updated_code": self.original_code})
        result = self.apply(fix=unchanged)
        self.assertEqual(result.status, "error")

    def test_missing_initial_fingerprint_prevents_write(self) -> None:
        result = self.apply(original_content_hash=None)
        self.assertEqual(result.status, "error")
        self.assertIn("fingerprint", result.warnings[0])


class ApplyPathBoundaryTests(TemporaryApplicationRepository):
    def test_target_outside_repository_is_rejected(self) -> None:
        outside = self.base / "outside.py"
        outside.write_text(self.original_code, encoding="utf-8")
        result = self.apply(
            target_file=str(outside),
            original_content_hash=content_fingerprint(outside.read_bytes()),
        )
        self.assertEqual(result.status, "error")
        self.assertIn("inside repository", result.warnings[0])

    def test_symlink_target_is_rejected_when_supported(self) -> None:
        real_target = self.repository / "src/real.py"
        real_target.write_text(self.original_code, encoding="utf-8")
        link = self.repository / "src/linked.py"
        try:
            os.symlink(real_target, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"File symlinks are unavailable: {error}")
        result = self.apply(
            target_file="src/linked.py",
            original_content_hash=content_fingerprint(real_target.read_bytes()),
        )
        self.assertEqual(result.status, "error")
        self.assertIn("symlink", result.warnings[0])
        self.assertEqual(real_target.read_text(encoding="utf-8"), self.original_code)

    def test_missing_target_is_rejected(self) -> None:
        result = self.apply(target_file="src/missing.py")
        self.assertEqual(result.status, "error")
        self.assertIn("does not exist", result.warnings[0])

    def test_invalid_repository_is_rejected(self) -> None:
        result = self.apply(repository_root=str(self.base / "missing-repo"))
        self.assertEqual(result.status, "error")
        self.assertIn("Repository root", result.warnings[0])

    def test_directory_target_is_rejected(self) -> None:
        result = self.apply(target_file="src")
        self.assertEqual(result.status, "error")
        self.assertIn("regular file", result.warnings[0])


class StaleTargetTests(TemporaryApplicationRepository):
    def test_changed_bytes_are_stale_and_never_overwritten(self) -> None:
        changed = "def add(a, b):\n    return 42\n"
        self.target.write_text(changed, encoding="utf-8")
        result = self.apply()
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.target.read_text(encoding="utf-8"), changed)

    def test_mismatched_hash_is_stale_even_when_text_is_unchanged(self) -> None:
        result = self.apply(original_content_hash="0" * 64)
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.original_code)

    def test_reviewed_content_mismatch_is_stale_even_when_hash_matches(self) -> None:
        result = self.apply(original_code="VALUE = 1\n")
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.original_code)


class AtomicApplicationTests(TemporaryApplicationRepository):
    def test_verified_unchanged_target_is_applied(self) -> None:
        result = self.apply()
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.target_file, "src/calculator.py")
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.updated_code)

    def test_atomic_temp_file_is_created_in_target_directory(self) -> None:
        with patch(
            "fix_application.tempfile.mkstemp",
            wraps=tempfile.mkstemp,
        ) as mkstemp_mock:
            result = self.apply()
        self.assertEqual(result.status, "applied")
        self.assertEqual(
            Path(mkstemp_mock.call_args.kwargs["dir"]).resolve(),
            self.target.parent.resolve(),
        )

    def test_os_replace_is_used_for_final_write(self) -> None:
        with patch("fix_application.os.replace", wraps=os.replace) as replace_mock:
            result = self.apply()
        self.assertEqual(result.status, "applied")
        replace_mock.assert_called_once()

    def test_write_failure_leaves_original_unchanged(self) -> None:
        before = self.target.read_bytes()
        with patch("fix_application.os.fsync", side_effect=OSError("disk full")):
            result = self.apply()
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(list(self.target.parent.glob(".calculator.py.tmp-*")), [])

    def test_change_during_atomic_preparation_is_stale(self) -> None:
        changed = "def add(a, b):\n    return 42\n"
        actual_fsync = os.fsync

        def sync_then_change(descriptor):
            actual_fsync(descriptor)
            self.target.write_text(changed, encoding="utf-8")

        with patch("fix_application.os.fsync", side_effect=sync_then_change):
            result = self.apply()
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.target.read_text(encoding="utf-8"), changed)
        self.assertEqual(list(self.target.parent.glob(".calculator.py.tmp-*")), [])

    def test_candidate_exact_bytes_are_written_and_confirmed(self) -> None:
        result = self.apply()
        self.assertEqual(self.target.read_bytes(), self.updated_code.encode("utf-8"))
        self.assertEqual(result.bytes_written, len(self.target.read_bytes()))

    def test_post_write_mismatch_is_reported_without_rollback(self) -> None:
        actual_replace = os.replace

        def replace_then_tamper(source, destination):
            actual_replace(source, destination)
            Path(destination).write_text("TAMPERED\n", encoding="utf-8")

        with patch("fix_application.os.replace", side_effect=replace_then_tamper):
            result = self.apply()
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_text(encoding="utf-8"), "TAMPERED\n")
        self.assertIn("no automatic rollback", result.warnings[0])

    def test_success_leaves_no_backup_or_temp_artifacts(self) -> None:
        result = self.apply()
        self.assertEqual(result.status, "applied")
        self.assertEqual(list(self.repository.rglob("*.bak")), [])
        self.assertEqual(list(self.target.parent.glob(".calculator.py.tmp-*")), [])

    def test_application_never_runs_git_commands(self) -> None:
        with patch.object(subprocess, "run") as subprocess_run:
            result = self.apply()
        self.assertEqual(result.status, "applied")
        subprocess_run.assert_not_called()

    def test_application_never_calls_llm(self) -> None:
        with patch("agent.ChatOpenAI") as chat_openai:
            result = self.apply()
        self.assertEqual(result.status, "applied")
        chat_openai.assert_not_called()


class EncodingAndNewlineTests(TemporaryApplicationRepository):
    def test_utf8_source_is_written_safely(self) -> None:
        updated = "def greeting():\n    return '你好'\n"
        result = self.apply(fix=self.fix.model_copy(update={"updated_code": updated}))
        self.assertEqual(result.status, "applied")
        self.assertEqual(self.target.read_text(encoding="utf-8"), updated)

    def test_encoding_failure_leaves_original_bytes_unchanged(self) -> None:
        original = "# coding: latin-1\nNAME = 'café'\n"
        original_bytes = original.encode("latin-1")
        self.target.write_bytes(original_bytes)
        updated = "# coding: latin-1\nNAME = '🙂'\n"
        result = self.apply(
            original_code=original,
            original_content_hash=content_fingerprint(original_bytes),
            fix=self.fix.model_copy(update={"updated_code": updated}),
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), original_bytes)

    def test_changed_encoding_cookie_is_rejected(self) -> None:
        updated = "# coding: latin-1\ndef add(a, b):\n    return a + b\n"
        result = self.apply(fix=self.fix.model_copy(update={"updated_code": updated}))
        self.assertEqual(result.status, "error")
        self.assertIn("encoding declaration", result.warnings[0])

    def test_lf_newlines_are_preserved(self) -> None:
        result = self.apply()
        self.assertEqual(result.status, "applied")
        self.assertNotIn(b"\r\n", self.target.read_bytes())

    def test_crlf_newlines_are_preserved(self) -> None:
        original_bytes = self.original_code.replace("\n", "\r\n").encode("utf-8")
        self.target.write_bytes(original_bytes)
        result = self.apply(
            original_content_hash=content_fingerprint(original_bytes)
        )
        self.assertEqual(result.status, "applied")
        written = self.target.read_bytes()
        self.assertIn(b"\r\n", written)
        self.assertNotIn(b"\n", written.replace(b"\r\n", b""))

    def test_mixed_newlines_are_refused(self) -> None:
        mixed = b"def add(a, b):\r\n    return a - b\n"
        self.target.write_bytes(mixed)
        result = self.apply(
            original_content_hash=content_fingerprint(mixed)
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(self.target.read_bytes(), mixed)
        self.assertIn("mixed newline", result.warnings[0])


@unittest.skipUnless(
    importlib.util.find_spec("pytest") is not None,
    "pytest is not installed in the current interpreter",
)
class ApplyIntegrationTests(TemporaryApplicationRepository):
    def setUp(self) -> None:
        super().setUp()
        test_file = self.repository / "tests/test_calculator.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            "from src.calculator import add\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n",
            encoding="utf-8",
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

    def verify_candidate(self) -> FixVerificationResult:
        with patch(
            "fix_verification.analyze_code",
            return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
        ):
            return verify_candidate_fix(
                str(self.repository),
                self.context,
                "src/calculator.py",
                self.updated_code,
                self.verification.patch,
            )

    def test_real_temporary_verification_then_apply(self) -> None:
        verification = self.verify_candidate()
        self.assertEqual(verification.status, "verified")
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.original_code)
        result = self.apply(verification=verification)
        self.assertEqual(result.status, "applied")
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.updated_code)

    def test_stale_change_after_verification_is_never_overwritten(self) -> None:
        verification = self.verify_candidate()
        changed = "def add(a, b):\n    return 42\n"
        self.target.write_text(changed, encoding="utf-8")
        result = self.apply(verification=verification)
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.target.read_text(encoding="utf-8"), changed)


if __name__ == "__main__":
    unittest.main()
