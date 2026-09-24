import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent import (
    CodeReview,
    FindingCategory,
    OverallRating,
    ReviewFinding,
    Severity,
    build_repository_context_node,
    main,
    review_code,
)
from repository_context import (
    RepoContext,
    RepositoryFile,
    build_repository_context,
    extract_python_imports,
)
from static_analysis import StaticAnalysisResult


def successful_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.NEEDS_WORK,
        summary="A bounded test review.",
        findings=[
            ReviewFinding(
                category=FindingCategory.BUG,
                severity=Severity.HIGH,
                title="Example issue",
                description="An example issue for graph tests.",
                line_number=1,
                suggestion="Address the example issue.",
            )
        ],
    )


def invalid_review() -> CodeReview:
    review = successful_review()
    return review.model_copy(update={"overall_rating": OverallRating.GOOD})


class TemporaryRepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name)
        self.repository = self.workspace / "repo"
        self.repository.mkdir()

    def write_repository_file(self, relative_path: str, content: str = "") -> Path:
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class RepositoryDiscoveryTests(TemporaryRepositoryTestCase):
    def test_basic_file_tree_is_sorted_and_deterministic(self) -> None:
        self.write_repository_file("utils.py", "VALUE = 1\n")
        self.write_repository_file("app.py", "return_value = 42\n")
        self.write_repository_file("notes.txt", "not included")

        first = build_repository_context(str(self.repository), "app.py")
        second = build_repository_context(str(self.repository), "app.py")

        self.assertEqual(first.file_tree, ["app.py", "utils.py"])
        self.assertEqual(first, second)

    def test_excluded_directories_do_not_enter_file_tree(self) -> None:
        self.write_repository_file("app.py", "return_value = 42\n")
        for directory in (
            ".git",
            ".venv",
            "__pycache__",
            "node_modules",
            "build",
        ):
            self.write_repository_file(f"{directory}/hidden.py", "SECRET = True\n")

        context = build_repository_context(str(self.repository), "app.py")

        self.assertEqual(context.file_tree, ["app.py"])

    def test_tree_entry_limit_is_reported(self) -> None:
        self.write_repository_file("target.py", "VALUE = 1\n")
        for index in range(5):
            self.write_repository_file(f"module_{index}.py", "VALUE = 1\n")

        with patch("repository_context.MAX_TREE_ENTRIES", 3):
            context = build_repository_context(str(self.repository), "target.py")

        self.assertEqual(len(context.file_tree), 3)
        self.assertTrue(
            any("MAX_TREE_ENTRIES=3" in warning for warning in context.warnings)
        )

    def test_path_traversal_and_absolute_escape_are_rejected(self) -> None:
        outside = self.workspace / "outside.py"
        outside.write_text("SECRET = True\n", encoding="utf-8")
        self.write_repository_file("target.py", "VALUE = 1\n")

        for target in ("../outside.py", str(outside)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_repository_context(str(self.repository), target)

    def test_symlink_escape_is_rejected_when_supported(self) -> None:
        outside = self.workspace / "outside.py"
        outside.write_text("SECRET = True\n", encoding="utf-8")
        link = self.repository / "linked.py"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"File symlinks are unavailable: {error}")

        with self.assertRaises(ValueError):
            build_repository_context(str(self.repository), "linked.py")


class ImportResolutionTests(TemporaryRepositoryTestCase):
    def test_import_extraction_is_normalized_without_execution(self) -> None:
        imports = extract_python_imports(
            "import package.module\n"
            "from . import models\n"
            "from ..utils import security\n"
        )

        normalized = [
            (item.level, item.module, item.names)
            for item in imports
        ]
        self.assertEqual(
            normalized,
            [
                (0, "package.module", ()),
                (1, "", ("models",)),
                (2, "utils", ("security",)),
            ],
        )

    def test_direct_local_import_is_included(self) -> None:
        self.write_repository_file("app.py", "from utils import helper\n")
        self.write_repository_file("utils.py", "def helper():\n    return 42\n")

        context = build_repository_context(str(self.repository), "app.py")

        self.assertEqual(
            [(item.path, item.relationship) for item in context.related_files],
            [("utils.py", "local_import")],
        )

    def test_relative_import_is_resolved_from_target_package(self) -> None:
        self.write_repository_file("package/__init__.py")
        self.write_repository_file(
            "package/service.py",
            "from .models import User\n",
        )
        self.write_repository_file("package/models.py", "class User:\n    pass\n")

        context = build_repository_context(
            str(self.repository),
            "package/service.py",
        )

        self.assertEqual(
            [item.path for item in context.related_files],
            ["package/models.py"],
        )

    def test_src_layout_absolute_import_is_resolved(self) -> None:
        self.write_repository_file("src/package/__init__.py")
        self.write_repository_file(
            "src/package/service.py",
            "from package.models import User\n",
        )
        self.write_repository_file(
            "src/package/models.py",
            "class User:\n    pass\n",
        )

        context = build_repository_context(
            str(self.repository),
            "src/package/service.py",
        )

        self.assertEqual(
            [item.path for item in context.related_files],
            ["src/package/models.py"],
        )

    def test_parent_relative_import_resolves_submodule(self) -> None:
        self.write_repository_file("src/users/__init__.py")
        self.write_repository_file(
            "src/users/service.py",
            "from ..utils import security\n",
        )
        self.write_repository_file("src/utils/__init__.py")
        self.write_repository_file("src/utils/security.py", "SAFE = True\n")

        context = build_repository_context(
            str(self.repository),
            "src/users/service.py",
        )

        self.assertIn(
            "src/utils/security.py",
            [item.path for item in context.related_files],
        )

    def test_duplicate_imports_produce_one_related_file(self) -> None:
        self.write_repository_file(
            "app.py",
            "import utils\nfrom utils import helper\n",
        )
        self.write_repository_file("utils.py", "def helper():\n    return 42\n")

        context = build_repository_context(str(self.repository), "app.py")

        self.assertEqual([item.path for item in context.related_files], ["utils.py"])

    def test_target_file_is_not_repeated_as_related_context(self) -> None:
        self.write_repository_file("app.py", "import app\n")

        context = build_repository_context(str(self.repository), "app.py")

        self.assertEqual(context.related_files, [])


class RelatedTestDiscoveryTests(TemporaryRepositoryTestCase):
    def test_related_test_is_discovered_for_src_target(self) -> None:
        self.write_repository_file("src/service.py", "VALUE = 1\n")
        self.write_repository_file(
            "tests/test_service.py",
            "def test_service():\n    assert True\n",
        )

        context = build_repository_context(str(self.repository), "src/service.py")

        self.assertEqual(
            [(item.path, item.relationship) for item in context.related_files],
            [("tests/test_service.py", "test")],
        )

    def test_related_test_count_is_limited(self) -> None:
        self.write_repository_file("src/service.py", "VALUE = 1\n")
        for relative_path in (
            "tests/test_service.py",
            "tests/service_test.py",
            "tests/unit/test_service.py",
            "tests/users/test_service.py",
        ):
            self.write_repository_file(relative_path, "VALUE = 1\n")

        context = build_repository_context(str(self.repository), "src/service.py")

        test_files = [
            item for item in context.related_files if item.relationship == "test"
        ]
        self.assertEqual(len(test_files), 3)
        self.assertTrue(
            any(
                "MAX_RELATED_TEST_FILES=3" in warning
                for warning in context.warnings
            )
        )


class ContextBudgetTests(TemporaryRepositoryTestCase):
    def test_single_file_content_limit_adds_warning(self) -> None:
        self.write_repository_file("app.py", "import dependency\n")
        self.write_repository_file("dependency.py", "X" * 100)

        with patch("repository_context.MAX_FILE_CHARS", 20):
            context = build_repository_context(str(self.repository), "app.py")

        self.assertEqual(len(context.related_files[0].content), 20)
        self.assertTrue(
            any("MAX_FILE_CHARS=20" in warning for warning in context.warnings)
        )

    def test_total_context_budget_is_enforced(self) -> None:
        self.write_repository_file("app.py", "import alpha\nimport beta\n")
        self.write_repository_file("alpha.py", "A" * 40)
        self.write_repository_file("beta.py", "B" * 40)

        with (
            patch("repository_context.MAX_FILE_CHARS", 50),
            patch("repository_context.MAX_TOTAL_CONTEXT_CHARS", 25),
        ):
            context = build_repository_context(str(self.repository), "app.py")

        total_chars = sum(len(item.content) for item in context.related_files)
        self.assertLessEqual(total_chars, 25)
        self.assertTrue(
            any("MAX_TOTAL_CONTEXT_CHARS" in warning for warning in context.warnings)
        )

    def test_related_file_count_limit_preserves_import_priority(self) -> None:
        imports = "\n".join(f"import module_{index}" for index in range(5))
        self.write_repository_file("app.py", f"{imports}\n")
        for index in range(5):
            self.write_repository_file(f"module_{index}.py", "VALUE = 1\n")
        self.write_repository_file("tests/test_app.py", "VALUE = 1\n")

        with patch("repository_context.MAX_RELATED_FILES", 3):
            context = build_repository_context(str(self.repository), "app.py")

        self.assertEqual(len(context.related_files), 3)
        self.assertTrue(
            all(item.relationship == "local_import" for item in context.related_files)
        )
        self.assertTrue(
            any("MAX_RELATED_FILES=3" in warning for warning in context.warnings)
        )


class RepositoryAgentIntegrationTests(TemporaryRepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.target = self.write_repository_file("app.py", "return_value = 42\n")

    def test_non_python_repository_mode_returns_warning_without_scan(self) -> None:
        with patch("agent.build_repository_context") as context_builder:
            result = build_repository_context_node(
                {
                    "language": "javascript",
                    "repository_root": str(self.repository),
                    "target_file": "app.js",
                }
            )

        context_builder.assert_not_called()
        context = result["repository_context"]
        self.assertIsInstance(context, RepoContext)
        self.assertEqual(context.related_files, [])
        self.assertTrue(any("supports Python" in item for item in context.warnings))

    def test_cross_file_optional_contract_reaches_repository_prompt_only(self) -> None:
        service_source = (
            "from users import repository\n\n"
            "def user_email(user_id: int) -> str:\n"
            "    user = repository.get_user(user_id)\n"
            "    return user.email\n"
        )
        self.write_repository_file("src/users/__init__.py")
        self.write_repository_file("src/users/service.py", service_source)
        self.write_repository_file(
            "src/users/repository.py",
            "def get_user(user_id: int) -> User | None:\n"
            "    ...\n",
        )
        self.write_repository_file(
            "tests/users/test_service.py",
            "def test_user_email():\n"
            "    assert True\n",
        )

        context = build_repository_context(
            str(self.repository),
            "src/users/service.py",
        )

        self.assertEqual(context.target_file, "src/users/service.py")
        relationships = {
            related.path: related.relationship
            for related in context.related_files
        }
        self.assertEqual(
            relationships["src/users/repository.py"],
            "local_import",
        )
        self.assertEqual(
            relationships["tests/users/test_service.py"],
            "test",
        )

        with (
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.return_value = successful_review()

            review_code(
                service_source,
                repository_root=str(self.repository),
                target_file="src/users/service.py",
            )
            repository_prompt = structured_llm.invoke.call_args.args[0][1].content

            review_code(service_source)
            single_file_prompt = structured_llm.invoke.call_args.args[0][1].content

        contract = "def get_user(user_id: int) -> User | None:"
        self.assertIn(contract, repository_prompt)
        self.assertIn("Related file: src/users/repository.py", repository_prompt)
        self.assertIn("Relationship: test", repository_prompt)
        self.assertNotIn(contract, single_file_prompt)
        self.assertNotIn("Repository context:", single_file_prompt)

    def test_inline_review_does_not_call_repository_builder(self) -> None:
        with (
            patch("agent.build_repository_context") as context_builder,
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.return_value = successful_review()

            result = review_code("return_value = 42")

        self.assertEqual(result, successful_review())
        context_builder.assert_not_called()

    def test_repository_context_node_runs_once_across_two_retries(self) -> None:
        repository_context = RepoContext(
            repository_root=str(self.repository),
            target_file="app.py",
            file_tree=["app.py"],
        )
        with (
            patch(
                "agent.build_repository_context",
                return_value=repository_context,
            ) as context_builder,
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ) as analyze_code,
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.side_effect = [
                invalid_review(),
                invalid_review(),
                successful_review(),
            ]

            result = review_code(
                "return_value = 42",
                repository_root=str(self.repository),
                target_file="app.py",
            )

        self.assertEqual(result, successful_review())
        context_builder.assert_called_once_with(str(self.repository), "app.py")
        analyze_code.assert_called_once_with("return_value = 42", "python")
        self.assertEqual(structured_llm.invoke.call_count, 3)

    def test_prompt_contains_repository_path_relationship_and_content(self) -> None:
        repository_context = RepoContext(
            repository_root=str(self.repository),
            target_file="app.py",
            file_tree=["app.py", "dependency.py"],
            related_files=[
                RepositoryFile(
                    path="dependency.py",
                    relationship="local_import",
                    content="CONTRACT = 42\n",
                )
            ],
        )
        with (
            patch(
                "agent.build_repository_context",
                return_value=repository_context,
            ),
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.return_value = successful_review()

            review_code(
                "return_value = 42",
                repository_root=str(self.repository),
                target_file="app.py",
            )

        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("Target: app.py", prompt)
        self.assertIn("Related file: dependency.py", prompt)
        self.assertIn("Relationship: local_import", prompt)
        self.assertIn("CONTRACT = 42", prompt)
        self.assertIn("Focus findings on the target file", prompt)

    def test_retry_prompt_keeps_repository_evidence_and_feedback(self) -> None:
        repository_context = RepoContext(
            repository_root=str(self.repository),
            target_file="app.py",
            related_files=[
                RepositoryFile(
                    path="tests/test_app.py",
                    relationship="test",
                    content="def test_app():\n    assert True\n",
                )
            ],
        )
        analysis = StaticAnalysisResult(
            tool_errors=["synthetic static-analysis failure"]
        )
        with (
            patch(
                "agent.build_repository_context",
                return_value=repository_context,
            ),
            patch("agent.analyze_code", return_value=analysis),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.side_effect = [
                invalid_review(),
                successful_review(),
            ]

            review_code(
                "return_value = 42",
                repository_root=str(self.repository),
                target_file="app.py",
            )

        retry_prompt = structured_llm.invoke.call_args_list[1].args[0][1].content
        self.assertIn("Related file: tests/test_app.py", retry_prompt)
        self.assertIn("Relationship: test", retry_prompt)
        self.assertIn("synthetic static-analysis failure", retry_prompt)
        self.assertIn("previous review failed semantic validation", retry_prompt)

    def test_repository_arguments_must_be_provided_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            review_code(
                "return_value = 42",
                repository_root=str(self.repository),
            )


class RepositoryCliTests(TemporaryRepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.target = self.write_repository_file("app.py", "return_value = 42\n")

    @patch("agent.review_code")
    def test_file_without_repo_keeps_single_file_mode(self, mocked_review_code) -> None:
        mocked_review_code.return_value = successful_review()
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["agent.py", "--file", str(self.target)]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "return_value = 42\n",
            "python",
            repository_root=None,
            target_file=None,
        )

    def test_code_with_repo_is_rejected_by_argparse(self) -> None:
        stderr = io.StringIO()
        argv = ["agent.py", "--code", "return 42", "--repo", "."]

        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as captured,
        ):
            main()

        self.assertEqual(captured.exception.code, 2)
        self.assertIn("--repo can only be used with --file", stderr.getvalue())

    @patch("agent.review_code")
    def test_file_with_repo_enables_repository_mode(self, mocked_review_code) -> None:
        mocked_review_code.return_value = successful_review()
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            str(self.repository),
        ]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "return_value = 42\n",
            "python",
            repository_root=str(self.repository),
            target_file="app.py",
        )


if __name__ == "__main__":
    unittest.main()
