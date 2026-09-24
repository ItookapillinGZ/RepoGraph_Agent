import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from change_set import ChangeSetReview
from engineering_plan import EngineeringPlan, PlannedFileChange
from plan_application import (
    MAX_APPLICATION_BUNDLE_CHARS,
    PlanApplicationBundle,
    PlanApplicationError,
    PlanApplicationResult,
    apply_plan_application_bundle,
    build_plan_application_bundle,
    calculate_plan_application_digest,
    load_plan_application_bundle,
    save_plan_application_bundle,
)
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    PlanExecutionResult,
    PlanExecutionVerification,
    _one_file_diff,
)
from review_models import OverallRating


class TemporaryPlanApplicationRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.repository = self.base / "repo"
        self.source = self.repository / "src"
        self.source.mkdir(parents=True)
        self.alpha = self.source / "alpha.py"
        self.beta = self.source / "beta.py"
        self.removed = self.source / "removed.py"
        self.alpha_original = "VALUE = 1\n"
        self.beta_original = "def lookup(value):\n    return value\n"
        self.removed_original = "OBSOLETE = True\n"
        self.alpha.write_text(self.alpha_original, encoding="utf-8", newline="")
        self.beta.write_text(self.beta_original, encoding="utf-8", newline="")
        self.removed.write_text(self.removed_original, encoding="utf-8", newline="")

    def execution_result(
        self,
        changes: list[tuple[str, str, str | None]] | None = None,
        *,
        status: str = "verified",
        verification_status: str = "verified",
        rating: OverallRating = OverallRating.GOOD,
        validation_errors: list[str] | None = None,
    ) -> PlanExecutionResult:
        selected = changes or [
            ("src/alpha.py", "modify", "VALUE = 2\n"),
        ]
        planned = [
            PlannedFileChange(
                path=path,
                action=action,
                rationale=f"{action} {path}",
            )
            for path, action, _ in selected
        ]
        candidate_changes = [
            CandidateFileChange(path=path, action=action, content=content)
            for path, action, content in selected
        ]
        originals = {
            "src/alpha.py": self.alpha_original,
            "src/beta.py": self.beta_original,
            "src/removed.py": self.removed_original,
        }
        diff = "".join(
            _one_file_diff(
                path,
                action,
                "" if action == "add" else originals[path],
                "" if action == "delete" else (content or ""),
            )
            for path, action, content in sorted(selected)
        )
        return PlanExecutionResult(
            plan=EngineeringPlan(
                summary="Apply a bounded candidate.",
                files=planned,
            ),
            candidate=MultiFileCandidate(
                summary="Candidate implementation.",
                files=candidate_changes,
            ),
            status=status,
            diff_text=diff,
            verification=PlanExecutionVerification(status=verification_status),
            change_set_review=ChangeSetReview(
                overall_rating=rating,
                summary="Verified candidate review.",
                file_results=[],
            ),
            validation_errors=validation_errors or [],
        )

    def bundle(self, changes=None) -> PlanApplicationBundle:
        return build_plan_application_bundle(
            str(self.repository),
            self.execution_result(changes),
        )

    def snapshot(self) -> dict[str, bytes]:
        return {
            path.relative_to(self.repository).as_posix(): path.read_bytes()
            for path in self.repository.rglob("*.py")
        }

    def internal_artifacts(self) -> list[Path]:
        return list(self.repository.rglob(".*.plan-*-*"))


class ApplicationSchemaAndBundleTests(TemporaryPlanApplicationRepository):
    def test_bundle_and_result_forbid_extra_fields(self) -> None:
        bundle = self.bundle()
        with self.assertRaises(ValidationError):
            PlanApplicationBundle.model_validate(
                {**bundle.model_dump(), "unexpected": True}
            )
        with self.assertRaises(ValidationError):
            PlanApplicationResult.model_validate(
                {"status": "applied", "unexpected": True}
            )

    def test_verified_good_execution_builds_bundle(self) -> None:
        bundle = self.bundle()
        self.assertEqual(bundle.schema_version, 1)
        self.assertEqual(bundle.plan.files[0].path, "src/alpha.py")

    def test_ineligible_execution_results_are_rejected(self) -> None:
        cases = [
            {"status": "failed"},
            {"status": "error"},
            {"verification_status": "failed"},
            {"rating": OverallRating.NEEDS_WORK},
            {"validation_errors": ["bad candidate"]},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(PlanApplicationError):
                build_plan_application_bundle(
                    str(self.repository),
                    self.execution_result(**changes),
                )

    def test_missing_candidate_is_rejected(self) -> None:
        result = self.execution_result().model_copy(update={"candidate": None})
        with self.assertRaises(PlanApplicationError):
            build_plan_application_bundle(str(self.repository), result)

    def test_original_and_candidate_hashes_are_captured(self) -> None:
        bundle = self.bundle()
        self.assertEqual(
            bundle.expected_original_hashes["src/alpha.py"],
            __import__("hashlib").sha256(self.alpha.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            bundle.expected_candidate_hashes["src/alpha.py"],
            __import__("hashlib").sha256(b"VALUE = 2\n").hexdigest(),
        )

    def test_add_and_delete_use_null_hash_semantics(self) -> None:
        bundle = self.bundle(
            [
                ("src/new_file.py", "add", "NEW = True\n"),
                ("src/removed.py", "delete", None),
            ]
        )
        self.assertIsNone(bundle.expected_original_hashes["src/new_file.py"])
        self.assertIsNone(bundle.expected_candidate_hashes["src/removed.py"])

    def test_approval_digest_is_deterministic_and_candidate_sensitive(self) -> None:
        first = self.bundle()
        second = self.bundle()
        self.assertEqual(first.approval_digest, second.approval_digest)
        changed = first.model_copy(
            update={
                "candidate": first.candidate.model_copy(
                    update={"summary": "Different approved summary."}
                )
            }
        )
        self.assertNotEqual(
            calculate_plan_application_digest(changed),
            first.approval_digest,
        )

    def test_plan_candidate_and_diff_mismatches_are_rejected(self) -> None:
        result = self.execution_result()
        bad_candidate = result.candidate.model_copy(
            update={
                "files": [
                    CandidateFileChange(
                        path="src/beta.py",
                        action="modify",
                        content="def lookup(value):\n    return value.lower()\n",
                    )
                ]
            }
        )
        with self.assertRaises(PlanApplicationError):
            build_plan_application_bundle(
                str(self.repository),
                result.model_copy(update={"candidate": bad_candidate}),
            )
        with self.assertRaises(PlanApplicationError):
            build_plan_application_bundle(
                str(self.repository),
                result.model_copy(update={"diff_text": result.diff_text + "\n"}),
            )


class ApplicationPreflightTests(TemporaryPlanApplicationRepository):
    def test_approved_false_is_zero_write(self) -> None:
        bundle = self.bundle()
        before = self.snapshot()
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=False
        )
        self.assertEqual(result.status, "not_requested")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.internal_artifacts(), [])

    def test_digest_mismatch_is_rejected_without_writes(self) -> None:
        bundle = self.bundle().model_copy(
            update={
                "candidate": self.bundle().candidate.model_copy(
                    update={"summary": "Tampered"}
                )
            }
        )
        before = self.snapshot()
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "error")
        self.assertIn("digest", result.failure_reason)
        self.assertEqual(self.snapshot(), before)

    def test_stale_modify_is_rejected_without_writes(self) -> None:
        bundle = self.bundle()
        self.alpha.write_text("VALUE = 99\n", encoding="utf-8", newline="")
        before = self.snapshot()
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.snapshot(), before)

    def test_stale_delete_is_rejected_without_other_writes(self) -> None:
        bundle = self.bundle(
            [
                ("src/alpha.py", "modify", "VALUE = 2\n"),
                ("src/removed.py", "delete", None),
            ]
        )
        self.removed.write_text("CHANGED = True\n", encoding="utf-8", newline="")
        before = self.snapshot()
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "stale")
        self.assertEqual(self.snapshot(), before)

    def test_added_path_becoming_existing_is_stale(self) -> None:
        bundle = self.bundle([("src/new_file.py", "add", "NEW = True\n")])
        new_file = self.source / "new_file.py"
        new_file.write_text("USER = True\n", encoding="utf-8", newline="")
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "stale")
        self.assertEqual(new_file.read_text(encoding="utf-8"), "USER = True\n")

    def test_manually_tampered_hash_map_is_rejected(self) -> None:
        bundle = self.bundle()
        changed = bundle.model_copy(
            update={"expected_original_hashes": {"src/other.py": "0" * 64}}
        )
        payload = changed.model_dump(mode="json")
        payload["approval_digest"] = calculate_plan_application_digest(payload)
        changed = PlanApplicationBundle.model_validate(payload)
        result = apply_plan_application_bundle(
            str(self.repository), changed, approved=True
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(self.alpha.read_text(encoding="utf-8"), self.alpha_original)

    def test_model_copy_with_invalid_nested_schema_is_controlled(self) -> None:
        bundle = self.bundle().model_copy(update={"candidate": "invalid"})
        before = self.snapshot()
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "error")
        self.assertIn("schema", result.failure_reason)
        self.assertEqual(self.snapshot(), before)

    def test_added_encoding_cookie_conflict_is_rejected(self) -> None:
        result = self.execution_result(
            [("src/new_file.py", "add", "# coding: latin-1\nNAME = 'x'\n")]
        )
        with self.assertRaises(PlanApplicationError):
            build_plan_application_bundle(str(self.repository), result)

    def test_unsafe_paths_in_recomputed_bundle_are_rejected(self) -> None:
        original_bundle = self.bundle()
        for unsafe_path in (
            "../outside.py",
            "C:/outside.py",
            ".env",
            ".git/hooks/unsafe.py",
        ):
            with self.subTest(path=unsafe_path):
                plan = original_bundle.plan.model_copy(
                    update={
                        "files": [
                            original_bundle.plan.files[0].model_copy(
                                update={"path": unsafe_path}
                            )
                        ]
                    }
                )
                candidate = original_bundle.candidate.model_copy(
                    update={
                        "files": [
                            original_bundle.candidate.files[0].model_copy(
                                update={"path": unsafe_path}
                            )
                        ]
                    }
                )
                changed = original_bundle.model_copy(
                    update={
                        "plan": plan,
                        "candidate": candidate,
                        "expected_original_hashes": {
                            unsafe_path: next(
                                iter(original_bundle.expected_original_hashes.values())
                            )
                        },
                        "expected_candidate_hashes": {
                            unsafe_path: next(
                                iter(original_bundle.expected_candidate_hashes.values())
                            )
                        },
                    }
                )
                payload = changed.model_dump(mode="json")
                payload["approval_digest"] = calculate_plan_application_digest(payload)
                changed = PlanApplicationBundle.model_validate(payload)
                before = self.snapshot()
                result = apply_plan_application_bundle(
                    str(self.repository), changed, approved=True
                )
                self.assertEqual(result.status, "error")
                self.assertEqual(self.snapshot(), before)

    def test_symlink_in_manually_recomputed_bundle_is_rejected_when_supported(
        self,
    ) -> None:
        link = self.source / "linked.py"
        try:
            os.symlink(self.alpha, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"File symlinks are unavailable: {error}")
        original_bundle = self.bundle()
        path = "src/linked.py"
        plan = original_bundle.plan.model_copy(
            update={
                "files": [
                    original_bundle.plan.files[0].model_copy(update={"path": path})
                ]
            }
        )
        candidate = original_bundle.candidate.model_copy(
            update={
                "files": [
                    original_bundle.candidate.files[0].model_copy(
                        update={"path": path}
                    )
                ]
            }
        )
        changed = original_bundle.model_copy(
            update={
                "plan": plan,
                "candidate": candidate,
                "expected_original_hashes": {
                    path: original_bundle.expected_original_hashes["src/alpha.py"]
                },
                "expected_candidate_hashes": {
                    path: original_bundle.expected_candidate_hashes["src/alpha.py"]
                },
            }
        )
        payload = changed.model_dump(mode="json")
        payload["approval_digest"] = calculate_plan_application_digest(payload)
        result = apply_plan_application_bundle(
            str(self.repository),
            PlanApplicationBundle.model_validate(payload),
            approved=True,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(self.alpha.read_text(encoding="utf-8"), self.alpha_original)


class TransactionTests(TemporaryPlanApplicationRepository):
    def test_modify_add_delete_transaction(self) -> None:
        bundle = self.bundle(
            [
                ("src/removed.py", "delete", None),
                ("src/new_file.py", "add", "NEW = True\r\n"),
                ("src/alpha.py", "modify", "VALUE = 2\n"),
            ]
        )
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(
            result.applied_files,
            ["src/alpha.py", "src/new_file.py", "src/removed.py"],
        )
        self.assertEqual(self.alpha.read_bytes(), b"VALUE = 2\n")
        self.assertEqual((self.source / "new_file.py").read_bytes(), b"NEW = True\n")
        self.assertFalse(self.removed.exists())
        self.assertEqual(self.internal_artifacts(), [])

    def test_modify_preserves_crlf_and_mode(self) -> None:
        original = b"VALUE = 1\r\n"
        self.alpha.write_bytes(original)
        os.chmod(self.alpha, 0o640)
        bundle = self.bundle()
        before_mode = stat.S_IMODE(self.alpha.stat().st_mode)
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(self.alpha.read_bytes(), b"VALUE = 2\r\n")
        self.assertEqual(stat.S_IMODE(self.alpha.stat().st_mode), before_mode)

    def test_modify_preserves_declared_encoding(self) -> None:
        original = "# coding: latin-1\nNAME = 'cafe'\n".encode("latin-1")
        self.alpha.write_bytes(original)
        result = self.execution_result(
            [
                (
                    "src/alpha.py",
                    "modify",
                    "# coding: latin-1\nNAME = 'café'\n",
                )
            ]
        )
        result.diff_text = _one_file_diff(
            "src/alpha.py",
            "modify",
            original.decode("latin-1"),
            result.candidate.files[0].content,
        )
        bundle = build_plan_application_bundle(str(self.repository), result)
        applied = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(applied.status, "applied")
        self.assertEqual(
            self.alpha.read_bytes(),
            "# coding: latin-1\nNAME = 'café'\n".encode("latin-1"),
        )

    def test_staging_failure_causes_zero_target_writes(self) -> None:
        bundle = self.bundle(
            [
                ("src/alpha.py", "modify", "VALUE = 2\n"),
                ("src/beta.py", "modify", "def lookup(value):\n    return None\n"),
            ]
        )
        before = self.snapshot()
        with patch("plan_application.os.fsync", side_effect=OSError("disk full")):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "error")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.internal_artifacts(), [])

    def test_second_operation_failure_rolls_back_first(self) -> None:
        bundle = self.bundle(
            [
                ("src/alpha.py", "modify", "VALUE = 2\n"),
                ("src/beta.py", "modify", "def lookup(value):\n    return None\n"),
                ("src/removed.py", "delete", None),
            ]
        )
        before = self.snapshot()
        from plan_application import _commit_entry

        calls = 0

        def fail_second(entry):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected")
            return _commit_entry(entry)

        with patch("plan_application._commit_entry", side_effect=fail_second):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.rolled_back_files, ["src/alpha.py"])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.internal_artifacts(), [])

    def test_replace_that_succeeds_then_raises_is_still_rolled_back(self) -> None:
        bundle = self.bundle([("src/new_file.py", "add", "NEW = True\n")])
        actual_replace = os.replace

        def replace_then_raise(source, destination):
            actual_replace(source, destination)
            raise OSError("injected after replace")

        with patch("plan_application.os.replace", side_effect=replace_then_raise):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.rolled_back_files, ["src/new_file.py"])
        self.assertFalse((self.source / "new_file.py").exists())

    def test_final_operation_failure_rolls_back_all_prior_operations(self) -> None:
        bundle = self.bundle(
            [
                ("src/alpha.py", "modify", "VALUE = 2\n"),
                ("src/beta.py", "modify", "def lookup(value):\n    return None\n"),
                ("src/removed.py", "delete", None),
            ]
        )
        before = self.snapshot()
        from plan_application import _commit_entry

        calls = 0

        def fail_third(entry):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("injected")
            return _commit_entry(entry)

        with patch("plan_application._commit_entry", side_effect=fail_third):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(
            result.rolled_back_files,
            ["src/alpha.py", "src/beta.py"],
        )
        self.assertEqual(self.snapshot(), before)

    def test_post_apply_hash_failure_rolls_back_everything(self) -> None:
        bundle = self.bundle(
            [
                ("src/alpha.py", "modify", "VALUE = 2\n"),
                ("src/new_file.py", "add", "NEW = True\n"),
                ("src/removed.py", "delete", None),
            ]
        )
        before = self.snapshot()
        with patch(
            "plan_application._post_apply_errors",
            return_value=["injected mismatch"],
        ):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(
            result.rolled_back_files,
            ["src/alpha.py", "src/new_file.py", "src/removed.py"],
        )
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.source / "new_file.py").exists())

    def test_rollback_restores_original_mode(self) -> None:
        os.chmod(self.alpha, 0o640)
        before_mode = stat.S_IMODE(self.alpha.stat().st_mode)
        bundle = self.bundle()
        with patch(
            "plan_application._post_apply_errors",
            return_value=["injected mismatch"],
        ):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(stat.S_IMODE(self.alpha.stat().st_mode), before_mode)

    def test_cleanup_failure_is_a_noncritical_applied_warning(self) -> None:
        bundle = self.bundle()
        with patch("pathlib.Path.unlink", side_effect=OSError("cleanup failed")):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "applied")
        self.assertTrue(result.warnings)
        for artifact in self.internal_artifacts():
            artifact.unlink()

    def test_apply_boundary_calls_no_llm_git_shell_or_tests(self) -> None:
        bundle = self.bundle()
        with (
            patch("agent.ChatOpenAI") as chat_model,
            patch.object(subprocess, "run") as subprocess_run,
        ):
            result = apply_plan_application_bundle(
                str(self.repository), bundle, approved=True
            )
        self.assertEqual(result.status, "applied")
        chat_model.assert_not_called()
        subprocess_run.assert_not_called()


class PlanApplicationIntegrationTests(TemporaryPlanApplicationRepository):
    def test_verified_two_file_lookup_candidate_applies_exact_bytes(self) -> None:
        tests_directory = self.repository / "tests"
        tests_directory.mkdir()
        service_test = tests_directory / "test_service.py"
        service_test.write_text(
            "def test_lookup():\n    assert True\n",
            encoding="utf-8",
            newline="",
        )
        changes = [
            (
                "src/beta.py",
                "modify",
                "def lookup(value):\n    return value.casefold()\n",
            ),
            (
                "tests/test_service.py",
                "modify",
                "def test_lookup():\n    assert 'A'.casefold() == 'a'\n",
            ),
        ]
        plan = EngineeringPlan(
            summary="Make email lookup case-insensitive.",
            files=[
                PlannedFileChange(path=path, action=action, rationale="Lookup fix.")
                for path, action, _ in changes
            ],
        )
        candidate = MultiFileCandidate(
            summary="Normalize lookup and verify behavior.",
            files=[
                CandidateFileChange(path=path, action=action, content=content)
                for path, action, content in changes
            ],
        )
        originals = {
            "src/beta.py": self.beta_original,
            "tests/test_service.py": "def test_lookup():\n    assert True\n",
        }
        diff = "".join(
            _one_file_diff(path, action, originals[path], content)
            for path, action, content in sorted(changes)
        )
        execution = PlanExecutionResult(
            plan=plan,
            candidate=candidate,
            status="verified",
            diff_text=diff,
            verification=PlanExecutionVerification(status="verified"),
            change_set_review=ChangeSetReview(
                overall_rating=OverallRating.GOOD,
                summary="Verified lookup change.",
                file_results=[],
            ),
        )
        before = self.snapshot()
        bundle = build_plan_application_bundle(str(self.repository), execution)
        self.assertEqual(self.snapshot(), before)
        result = apply_plan_application_bundle(
            str(self.repository), bundle, approved=True
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(self.beta.read_bytes(), changes[0][2].encode("utf-8"))
        self.assertEqual(service_test.read_bytes(), changes[1][2].encode("utf-8"))


class BundlePersistenceTests(TemporaryPlanApplicationRepository):
    def test_bundle_round_trip_uses_stable_json(self) -> None:
        bundle = self.bundle()
        output = self.base / "approved.json"
        save_plan_application_bundle(
            str(output),
            bundle,
            repository_root=str(self.repository),
        )
        loaded = load_plan_application_bundle(str(output))
        self.assertEqual(loaded, bundle)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["approval_digest"], bundle.approval_digest)

    def test_bundle_inside_repository_is_rejected(self) -> None:
        with self.assertRaises(PlanApplicationError):
            save_plan_application_bundle(
                str(self.repository / "approved.json"),
                self.bundle(),
                repository_root=str(self.repository),
            )

    def test_bounded_bundle_read(self) -> None:
        output = self.base / "oversized.json"
        output.write_text("x" * (MAX_APPLICATION_BUNDLE_CHARS + 1), encoding="utf-8")
        with self.assertRaises(PlanApplicationError):
            load_plan_application_bundle(str(output))

    def test_serialized_bundle_has_no_temporary_workspace_fields(self) -> None:
        serialized = self.bundle().model_dump_json()
        self.assertNotIn("temporary_repository_root", serialized)
        self.assertNotIn("temporary_workspace_base", serialized)


if __name__ == "__main__":
    unittest.main()
