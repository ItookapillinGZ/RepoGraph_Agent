"""Schema and deterministic validation tests for EngineeringPlan."""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from engineering_plan import (
    MAX_PLAN_ASSUMPTIONS,
    MAX_PLAN_FILES,
    MAX_PLAN_RISKS,
    MAX_PLAN_TESTS,
    EngineeringPlan,
    PlannedFileChange,
    PlannedTest,
    validate_engineering_plan,
)
from repository_exploration import RepositoryExplorationResult


def make_plan(
    *changes: PlannedFileChange,
    tests: list[PlannedTest] | None = None,
) -> EngineeringPlan:
    return EngineeringPlan(
        summary="Implement the requested repository change.",
        files=list(changes),
        tests=tests or [],
    )


class EngineeringPlanSchemaTests(unittest.TestCase):
    def test_planned_file_change_schema_and_extra_forbid(self) -> None:
        change = PlannedFileChange(
            path="src/service.py",
            action="modify",
            rationale="Adjust lookup normalization.",
        )

        self.assertEqual(change.action, "modify")
        with self.assertRaises(ValidationError):
            PlannedFileChange.model_validate(
                {**change.model_dump(), "commands": ["pytest"]}
            )

    def test_planned_test_schema_and_extra_forbid(self) -> None:
        planned_test = PlannedTest(
            path="tests/test_service.py",
            purpose="Cover mixed-case email input.",
        )

        self.assertEqual(planned_test.path, "tests/test_service.py")
        with self.assertRaises(ValidationError):
            PlannedTest.model_validate(
                {**planned_test.model_dump(), "terminal_steps": ["pytest"]}
            )

    def test_engineering_plan_schema_forbids_commands(self) -> None:
        payload = make_plan(
            PlannedFileChange(
                path="src/service.py",
                action="modify",
                rationale="Change behavior.",
            )
        ).model_dump()
        payload["git_commands"] = ["git commit"]

        with self.assertRaises(ValidationError):
            EngineeringPlan.model_validate(payload)

    def test_plan_collection_budgets_are_schema_enforced(self) -> None:
        change = {
            "path": "src/service.py",
            "action": "modify",
            "rationale": "Change behavior.",
        }
        base = {
            "summary": "Bounded plan.",
            "files": [change],
            "tests": [],
            "risks": [],
            "assumptions": [],
        }

        for field, limit, item in (
            ("files", MAX_PLAN_FILES, change),
            ("tests", MAX_PLAN_TESTS, {"path": None, "purpose": "Verify."}),
            ("risks", MAX_PLAN_RISKS, "risk"),
            ("assumptions", MAX_PLAN_ASSUMPTIONS, "assumption"),
        ):
            payload = dict(base)
            payload[field] = [item] * (limit + 1)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                EngineeringPlan.model_validate(payload)


class EngineeringPlanValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "service.py").write_text(
            "def find_user(email):\n    return email\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "def find_by_email(email):\n    return email\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_service():\n    assert True\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def exploration(
        self,
        *,
        files_read: list[str] | None = None,
        search_result_files: list[str] | None = None,
        status: str = "completed",
    ) -> RepositoryExplorationResult:
        return RepositoryExplorationResult(
            status=status,
            files_read=files_read or [],
            search_result_files=search_result_files or [],
        )

    def errors(
        self,
        plan: EngineeringPlan,
        exploration: RepositoryExplorationResult | None = None,
    ) -> list[str]:
        return validate_engineering_plan(
            plan,
            str(self.root),
            exploration
            or self.exploration(files_read=["src/service.py"]),
        )

    def change(self, path: str, action: str) -> PlannedFileChange:
        return PlannedFileChange(
            path=path,
            action=action,
            rationale="Required by the task.",
        )

    def test_modify_existing_grounded_file_is_accepted(self) -> None:
        self.assertEqual(
            self.errors(make_plan(self.change("src/service.py", "modify"))),
            [],
        )

    def test_modify_missing_file_is_rejected(self) -> None:
        errors = self.errors(make_plan(self.change("src/missing.py", "modify")))

        self.assertTrue(any("does not exist" in error for error in errors))

    def test_delete_existing_file_is_accepted(self) -> None:
        self.assertEqual(
            self.errors(make_plan(self.change("src/service.py", "delete"))),
            [],
        )

    def test_delete_missing_file_is_rejected(self) -> None:
        errors = self.errors(make_plan(self.change("src/missing.py", "delete")))

        self.assertTrue(any("does not exist" in error for error in errors))

    def test_add_missing_file_with_existing_parent_is_accepted(self) -> None:
        self.assertEqual(
            self.errors(make_plan(self.change("src/new_service.py", "add"))),
            [],
        )

    def test_add_existing_file_is_rejected(self) -> None:
        errors = self.errors(make_plan(self.change("src/service.py", "add")))

        self.assertTrue(any("already exists" in error for error in errors))

    def test_absolute_and_parent_traversal_paths_are_rejected(self) -> None:
        for path in (str(self.root / "src" / "service.py"), "../service.py"):
            with self.subTest(path=path):
                errors = self.errors(make_plan(self.change(path, "modify")))
                self.assertTrue(any("invalid" in error for error in errors))

    def test_excluded_and_secret_paths_are_rejected(self) -> None:
        for path in (".git/config", ".venv/tool.py", ".env", "keys/app.pem"):
            with self.subTest(path=path):
                errors = self.errors(make_plan(self.change(path, "add")))
                self.assertTrue(any("not allowed" in error for error in errors))

    def test_symlink_escape_is_rejected(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside"
        outside.mkdir()
        self.addCleanup(outside.rmdir)
        link = self.root / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")

        errors = self.errors(make_plan(self.change("linked/new.py", "add")))

        self.assertTrue(any("Symbolic links" in error for error in errors))

    def test_duplicate_normalized_planned_paths_are_rejected(self) -> None:
        plan = make_plan(
            self.change("src/service.py", "modify"),
            self.change("src\\service.py", "delete"),
        )

        errors = self.errors(plan)

        self.assertTrue(any("Duplicate planned path" in error for error in errors))

    def test_duplicate_tests_are_rejected(self) -> None:
        planned_test = PlannedTest(
            path="tests/test_service.py",
            purpose="Cover mixed-case input.",
        )
        plan = make_plan(
            self.change("src/service.py", "modify"),
            tests=[planned_test, planned_test],
        )

        errors = self.errors(plan)

        self.assertIn("Duplicate planned test is not allowed.", errors)

    def test_missing_test_path_requires_matching_add_change(self) -> None:
        test = PlannedTest(path="tests/test_new.py", purpose="Cover the change.")

        rejected = self.errors(
            make_plan(self.change("src/service.py", "modify"), tests=[test])
        )
        accepted = self.errors(
            make_plan(
                self.change("src/service.py", "modify"),
                self.change("tests/test_new.py", "add"),
                tests=[test],
            )
        )

        self.assertTrue(any("test path" in error for error in rejected))
        self.assertEqual(accepted, [])

    def test_ungrounded_modify_is_rejected(self) -> None:
        errors = self.errors(
            make_plan(self.change("src/repository.py", "modify")),
            self.exploration(files_read=["src/service.py"]),
        )

        self.assertTrue(any("not grounded" in error for error in errors))

    def test_files_read_and_structured_search_results_ground_modifies(self) -> None:
        plan = make_plan(
            self.change("src/service.py", "modify"),
            self.change("src/repository.py", "modify"),
        )
        exploration = self.exploration(
            files_read=["src/service.py"],
            search_result_files=["src/repository.py"],
        )

        self.assertEqual(self.errors(plan, exploration), [])

    def test_unstructured_summary_does_not_ground_a_modify(self) -> None:
        exploration = self.exploration()
        exploration = exploration.model_copy(
            update={"summary": "Search mentioned src/repository.py:1:"}
        )

        errors = self.errors(
            make_plan(self.change("src/repository.py", "modify")),
            exploration,
        )

        self.assertTrue(any("not grounded" in error for error in errors))

    def test_exploration_error_cannot_return_a_reliable_plan(self) -> None:
        errors = self.errors(
            make_plan(self.change("src/service.py", "modify")),
            self.exploration(
                files_read=["src/service.py"],
                status="error",
            ),
        )

        self.assertTrue(any("exploration failed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
