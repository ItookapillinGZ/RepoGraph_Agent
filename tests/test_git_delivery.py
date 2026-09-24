import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from change_set import ChangeSetReview
from engineering_plan import EngineeringPlan, PlannedFileChange
from git_delivery import (
    MAX_COMMIT_MESSAGE_CHARS,
    LocalGitDeliveryResult,
    _DeliveryFailure,
    _git_environment,
    _run_git,
    create_local_git_delivery,
    render_local_git_delivery_result,
)
from plan_application import (
    PlanApplicationBundle,
    apply_plan_application_bundle,
    build_plan_application_bundle,
    calculate_plan_application_digest,
)
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    PlanExecutionResult,
    PlanExecutionVerification,
    _one_file_diff,
)
from review_models import OverallRating


class TemporaryGitDeliveryRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.repository = self.base / "repo"
        self.source = self.repository / "src"
        self.source.mkdir(parents=True)
        self.alpha = self.source / "alpha.py"
        self.removed = self.source / "removed.py"
        self.alpha_original = "VALUE = 1\n"
        self.removed_original = "OBSOLETE = True\n"
        self.alpha.write_text(self.alpha_original, encoding="utf-8", newline="")
        self.removed.write_text(
            self.removed_original,
            encoding="utf-8",
            newline="",
        )
        self.git("init", "-b", "main")
        self.git("config", "user.name", "RepoGraph Test")
        self.git("config", "user.email", "repograph@example.invalid")
        self.git("add", "--", "src/alpha.py", "src/removed.py")
        self.git("commit", "-m", "Initial")
        self.initial_head = self.git("rev-parse", "HEAD").strip()

    def git(
        self,
        *args: str,
        text: bool = True,
        check: bool = True,
    ) -> str | bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repository,
            capture_output=True,
            text=text,
            check=check,
        )
        return completed.stdout

    def execution_result(
        self,
        changes: list[tuple[str, str, str | None]],
    ) -> PlanExecutionResult:
        originals = {
            "src/alpha.py": (
                self.alpha.read_text(encoding="utf-8")
                if self.alpha.exists()
                else ""
            ),
            "src/removed.py": (
                self.removed.read_text(encoding="utf-8")
                if self.removed.exists()
                else ""
            ),
        }
        plan = EngineeringPlan(
            summary="Apply approved repository changes.",
            files=[
                PlannedFileChange(
                    path=path,
                    action=action,
                    rationale=f"{action} {path}",
                )
                for path, action, _ in changes
            ],
        )
        candidate = MultiFileCandidate(
            summary="Approved candidate.",
            files=[
                CandidateFileChange(path=path, action=action, content=content)
                for path, action, content in changes
            ],
        )
        diff_text = "".join(
            _one_file_diff(
                path,
                action,
                "" if action == "add" else originals[path],
                "" if action == "delete" else (content or ""),
            )
            for path, action, content in sorted(changes)
        )
        return PlanExecutionResult(
            plan=plan,
            candidate=candidate,
            status="verified",
            diff_text=diff_text,
            verification=PlanExecutionVerification(status="verified"),
            change_set_review=ChangeSetReview(
                overall_rating=OverallRating.GOOD,
                summary="Approved change set.",
                file_results=[],
            ),
        )

    def bundle(
        self,
        changes: list[tuple[str, str, str | None]] | None = None,
    ) -> PlanApplicationBundle:
        selected = changes or [("src/alpha.py", "modify", "VALUE = 2\n")]
        return build_plan_application_bundle(
            str(self.repository),
            self.execution_result(selected),
        )

    def apply(self, bundle: PlanApplicationBundle) -> None:
        result = apply_plan_application_bundle(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "applied")

    def index_bytes(self) -> bytes | None:
        value = str(self.git("rev-parse", "--git-path", "index")).strip()
        index = Path(value)
        if not index.is_absolute():
            index = self.repository / index
        return index.read_bytes() if index.exists() else None

    def branch_ref(self, branch: str) -> str | None:
        completed = subprocess.run(
            ["git", "show-ref", "--verify", "--hash", f"refs/heads/{branch}"],
            cwd=self.repository,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None


class LocalGitDeliverySchemaTests(unittest.TestCase):
    def test_result_schema_is_strict(self) -> None:
        with self.assertRaises(ValidationError):
            LocalGitDeliveryResult.model_validate(
                {"status": "created", "unexpected": True}
            )

    def test_renderer_is_deterministic(self) -> None:
        result = LocalGitDeliveryResult(
            status="created",
            branch_name="repograph/example",
            commit_sha="a" * 40,
            base_sha="b" * 40,
            approval_digest="c" * 64,
            committed_files=["src/a.py"],
        )
        rendered = render_local_git_delivery_result(result)
        self.assertIn("Status: created", rendered)
        self.assertIn("Branch: repograph/example", rendered)
        self.assertIn("- src/a.py", rendered)


class LocalGitDeliveryPreflightTests(TemporaryGitDeliveryRepository):
    def test_approved_false_runs_no_git_and_creates_no_branch(self) -> None:
        bundle = self.bundle()
        with patch("git_delivery._run_git") as run_git:
            result = create_local_git_delivery(
                str(self.repository),
                bundle,
                approved=False,
            )
        self.assertEqual(result.status, "not_requested")
        run_git.assert_not_called()
        self.assertIsNone(
            self.branch_ref(f"repograph/{bundle.approval_digest[:12]}")
        )

    def test_non_git_repository_and_nested_root_are_rejected(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        plain = self.base / "plain"
        plain.mkdir()
        result = create_local_git_delivery(str(plain), bundle, approved=True)
        self.assertEqual(result.status, "error")

        nested = self.repository / "src"
        result = create_local_git_delivery(str(nested), bundle, approved=True)
        self.assertEqual(result.status, "error")
        self.assertIn("worktree root", result.failure_reason or "")

    def test_unborn_head_is_rejected(self) -> None:
        unborn = self.base / "unborn"
        unborn.mkdir()
        completed = subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=unborn,
            capture_output=True,
            check=True,
        )
        self.assertEqual(completed.returncode, 0)
        bundle = self.bundle()
        result = create_local_git_delivery(str(unborn), bundle, approved=True)
        self.assertEqual(result.status, "error")
        self.assertIn("HEAD commit", result.failure_reason or "")

    def test_invalid_bundle_and_digest_are_controlled(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        invalid = bundle.model_copy(
            update={"candidate": {"not": "a candidate"}}
        )
        result = create_local_git_delivery(
            str(self.repository),
            invalid,
            approved=True,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("schema", (result.failure_reason or "").casefold())

        changed = bundle.model_copy(
            update={
                "candidate": bundle.candidate.model_copy(
                    update={"summary": "Tampered"}
                )
            }
        )
        result = create_local_git_delivery(
            str(self.repository),
            changed,
            approved=True,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("digest", (result.failure_reason or "").casefold())


class LocalGitDeliveryMetadataTests(TemporaryGitDeliveryRepository):
    def test_default_branch_and_commit_message_are_deterministic(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        expected_branch = f"repograph/{bundle.approval_digest[:12]}"
        self.assertEqual(result.status, "created")
        self.assertEqual(result.branch_name, expected_branch)
        message = str(
            self.git("log", "-1", "--format=%B", expected_branch)
        )
        self.assertIn("RepoGraph: apply approved repository plan", message)
        self.assertIn(f"Approval-Digest: {bundle.approval_digest}", message)

    def test_explicit_branch_and_message_are_used(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
            branch_name="feature/approved-change",
            commit_message="Apply exact approved change",
        )
        self.assertEqual(result.status, "created")
        self.assertEqual(result.branch_name, "feature/approved-change")
        message = str(
            self.git("log", "-1", "--format=%B", "feature/approved-change")
        )
        self.assertIn("Apply exact approved change", message)

    def test_invalid_branch_names_are_rejected_without_ref_creation(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        for branch in ("HEAD", "refs/heads/unsafe", "-", "bad..name", " bad"):
            with self.subTest(branch=branch):
                result = create_local_git_delivery(
                    str(self.repository),
                    bundle,
                    approved=True,
                    branch_name=branch,
                )
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.commit_sha)

    def test_existing_branch_is_a_conflict_and_is_not_modified(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        self.git("branch", "already-there", self.initial_head)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
            branch_name="already-there",
        )
        self.assertEqual(result.status, "conflict")
        self.assertEqual(self.branch_ref("already-there"), self.initial_head)

    def test_commit_message_nul_and_budget_are_rejected(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        for message in ("bad\x00message", "x" * MAX_COMMIT_MESSAGE_CHARS):
            with self.subTest(length=len(message)):
                result = create_local_git_delivery(
                    str(self.repository),
                    bundle,
                    approved=True,
                    branch_name=f"message-{len(message)}",
                    commit_message=message,
                )
                self.assertEqual(result.status, "error")

    def test_missing_identity_is_controlled(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        self.git("config", "user.name", "")
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("user.name", result.failure_reason or "")

        self.git("config", "user.name", "RepoGraph Test")
        self.git("config", "user.email", "")
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("user.email", result.failure_reason or "")


class LocalGitDeliveryStateTests(TemporaryGitDeliveryRepository):
    def test_not_yet_applied_and_changed_after_apply_are_stale(self) -> None:
        bundle = self.bundle()
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "stale")

        self.apply(bundle)
        self.alpha.write_text("VALUE = 3\n", encoding="utf-8", newline="")
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "stale")
        self.assertIsNone(
            self.branch_ref(f"repograph/{bundle.approval_digest[:12]}")
        )

    def test_pre_existing_modify_change_relative_to_head_is_conflict(self) -> None:
        self.alpha.write_text("VALUE = 10\n", encoding="utf-8", newline="")
        bundle = self.bundle(
            [("src/alpha.py", "modify", "VALUE = 11\n")]
        )
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "conflict")
        self.assertIn("pre-existing", result.failure_reason or "")

    def test_pre_existing_delete_change_relative_to_head_is_conflict(self) -> None:
        self.removed.write_text(
            "OBSOLETE = 'user'\n",
            encoding="utf-8",
            newline="",
        )
        bundle = self.bundle([("src/removed.py", "delete", None)])
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "conflict")

    def test_add_that_already_exists_in_head_is_conflict(self) -> None:
        self.alpha.unlink()
        bundle = self.bundle([("src/alpha.py", "add", "VALUE = 2\n")])
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "conflict")

    def test_digest_recomputed_with_inconsistent_candidate_hash_is_stale(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        payload = bundle.model_dump(mode="python")
        payload["expected_candidate_hashes"]["src/alpha.py"] = hashlib.sha256(
            b"unrelated"
        ).hexdigest()
        payload["approval_digest"] = calculate_plan_application_digest(payload)
        changed = PlanApplicationBundle.model_validate(payload)
        result = create_local_git_delivery(
            str(self.repository),
            changed,
            approved=True,
        )
        self.assertEqual(result.status, "stale")

    def test_recomputed_digest_with_tampered_diff_body_is_error(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        payload = bundle.model_dump(mode="python")
        payload["diff_text"] = payload["diff_text"].replace(
            "+VALUE = 2",
            "+VALUE = 999",
        )
        payload["approval_digest"] = calculate_plan_application_digest(payload)
        changed = PlanApplicationBundle.model_validate(payload)
        result = create_local_git_delivery(
            str(self.repository),
            changed,
            approved=True,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("diff text", (result.failure_reason or "").casefold())


class LocalGitDeliveryIntegrationTests(TemporaryGitDeliveryRepository):
    def test_modify_add_delete_commit_is_exact_and_preserves_user_state(self) -> None:
        unrelated = self.repository / "notes.txt"
        staged = self.repository / "staged.py"
        unrelated.write_text("user working change\n", encoding="utf-8")
        staged.write_text("STAGED = True\n", encoding="utf-8")
        self.git("add", "--", "staged.py")
        staged.write_text("STAGED = 'worktree'\n", encoding="utf-8")

        changes = [
            ("src/alpha.py", "modify", "VALUE = 2\n"),
            ("src/new_file.py", "add", "NEW = True\n"),
            ("src/removed.py", "delete", None),
        ]
        bundle = self.bundle(changes)
        self.apply(bundle)

        status_before = self.git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        index_before = self.index_bytes()
        alpha_before = self.alpha.read_bytes()
        new_before = (self.source / "new_file.py").read_bytes()
        notes_before = unrelated.read_bytes()

        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )

        self.assertEqual(result.status, "created", result)
        self.assertEqual(result.base_sha, self.initial_head)
        self.assertEqual(
            self.git("rev-parse", "HEAD").strip(),
            self.initial_head,
        )
        self.assertEqual(
            self.git("symbolic-ref", "--short", "HEAD").strip(),
            "main",
        )
        self.assertEqual(self.index_bytes(), index_before)
        self.assertEqual(
            self.git("status", "--porcelain=v1", "--untracked-files=all"),
            status_before,
        )
        self.assertEqual(self.alpha.read_bytes(), alpha_before)
        self.assertEqual((self.source / "new_file.py").read_bytes(), new_before)
        self.assertEqual(unrelated.read_bytes(), notes_before)

        branch = result.branch_name or ""
        self.assertEqual(
            self.git("rev-parse", f"{branch}^").strip(),
            self.initial_head,
        )
        self.assertEqual(
            self.git("show", f"{branch}:src/alpha.py", text=False),
            b"VALUE = 2\n",
        )
        self.assertEqual(
            self.git("show", f"{branch}:src/new_file.py", text=False),
            b"NEW = True\n",
        )
        deleted = subprocess.run(
            ["git", "cat-file", "-e", f"{branch}:src/removed.py"],
            cwd=self.repository,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(deleted.returncode, 0)
        committed_paths = set(
            str(
                self.git(
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    branch,
                )
            ).splitlines()
        )
        self.assertEqual(
            committed_paths,
            {"src/alpha.py", "src/new_file.py", "src/removed.py"},
        )
        self.assertNotIn("staged.py", committed_paths)
        self.assertNotIn("notes.txt", committed_paths)

    def test_unrelated_dirty_and_staged_files_are_allowed(self) -> None:
        (self.repository / "notes.txt").write_text(
            "unstaged\n",
            encoding="utf-8",
        )
        (self.repository / "other.py").write_text(
            "OTHER = True\n",
            encoding="utf-8",
        )
        self.git("add", "--", "other.py")
        bundle = self.bundle()
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "created")

    def test_commit_tree_does_not_execute_hooks(self) -> None:
        marker = self.repository / "hook-ran.txt"
        hooks = self.repository / "hooks"
        hooks.mkdir()
        hook = hooks / "pre-commit"
        hook.write_text(
            f"#!/bin/sh\nprintf ran > '{marker.as_posix()}'\n",
            encoding="utf-8",
            newline="",
        )
        try:
            os.chmod(hook, 0o755)
        except OSError:
            pass
        self.git("config", "core.hooksPath", str(hooks))
        bundle = self.bundle()
        self.apply(bundle)
        result = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "created")
        self.assertFalse(marker.exists())

    def test_post_verification_failure_rolls_back_only_created_ref(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        self.git("branch", "keep-me", self.initial_head)
        branch = f"repograph/{bundle.approval_digest[:12]}"
        with patch(
            "git_delivery._verify_commit",
            side_effect=RuntimeError("forced post verification failure"),
        ):
            result = create_local_git_delivery(
                str(self.repository),
                bundle,
                approved=True,
            )
        self.assertEqual(result.status, "error")
        self.assertIsNone(self.branch_ref(branch))
        self.assertEqual(self.branch_ref("keep-me"), self.initial_head)
        self.assertTrue(
            any("rolled back" in warning for warning in result.warnings)
        )

    def test_all_git_commands_are_local_allowlisted_plumbing(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        from git_delivery import _run_git as real_run_git

        with patch("git_delivery._run_git", wraps=real_run_git) as run_git:
            result = create_local_git_delivery(
                str(self.repository),
                bundle,
                approved=True,
                branch_name="audit/commands",
            )
        self.assertEqual(result.status, "created")
        commands = [call.args[1] for call in run_git.call_args_list]
        forbidden = {
            "add",
            "checkout",
            "commit",
            "fetch",
            "merge",
            "pull",
            "push",
            "reset",
            "stash",
            "switch",
        }
        self.assertFalse(forbidden & {args[0] for args in commands})
        self.assertNotIn(["add", "."], commands)
        self.assertNotIn(["add", "-A"], commands)
        isolated = [
            call
            for call in run_git.call_args_list
            if call.args[1][0] in {"read-tree", "update-index", "write-tree"}
        ]
        self.assertTrue(isolated)
        self.assertTrue(
            all("GIT_INDEX_FILE" in call.kwargs.get("env", {}) for call in isolated)
        )

    def test_branch_creation_race_never_overwrites_new_ref(self) -> None:
        bundle = self.bundle()
        self.apply(bundle)
        branch = "race/approved"
        full_ref = f"refs/heads/{branch}"
        from git_delivery import _run_git as real_run_git

        raced = False

        def run_with_race(repository_root, args, **kwargs):
            nonlocal raced
            if (
                not raced
                and args[0] == "update-ref"
                and len(args) == 4
                and args[1] == full_ref
            ):
                raced = True
                subprocess.run(
                    ["git", "update-ref", full_ref, self.initial_head],
                    cwd=self.repository,
                    capture_output=True,
                    check=True,
                )
            return real_run_git(repository_root, args, **kwargs)

        with patch("git_delivery._run_git", side_effect=run_with_race):
            result = create_local_git_delivery(
                str(self.repository),
                bundle,
                approved=True,
                branch_name=branch,
            )
        self.assertEqual(result.status, "conflict")
        self.assertEqual(self.branch_ref(branch), self.initial_head)


class GitSubprocessBoundaryTests(unittest.TestCase):
    def test_dangerous_git_environment_is_removed(self) -> None:
        dangerous = {
            "GIT_DIR": "elsewhere",
            "GIT_WORK_TREE": "elsewhere",
            "GIT_INDEX_FILE": "elsewhere",
            "GIT_OBJECT_DIRECTORY": "elsewhere",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": "elsewhere",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": "elsewhere",
        }
        with patch.dict(os.environ, dangerous, clear=False):
            environment = _git_environment()
        for key in dangerous:
            self.assertNotIn(key, environment)
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")

    def test_shell_is_false_and_output_is_bounded(self) -> None:
        completed = subprocess.CompletedProcess(
            ["git", "status"],
            0,
            stdout="x" * 30_000,
            stderr="",
        )
        with patch("git_delivery.subprocess.run", return_value=completed) as run:
            result = _run_git(".", ["status", "--porcelain=v1"])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertTrue(run.call_args.kwargs["text"])
        self.assertTrue(result.output_truncated)
        self.assertLessEqual(len(result.stdout), 20_000)

    def test_timeout_and_non_allowlisted_command_are_controlled(self) -> None:
        with (
            patch(
                "git_delivery.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["git", "status"], 30),
            ),
            self.assertRaises(_DeliveryFailure),
        ):
            _run_git(".", ["status", "--porcelain=v1"])
        with self.assertRaises(_DeliveryFailure):
            _run_git(".", ["push", "origin", "main"])
