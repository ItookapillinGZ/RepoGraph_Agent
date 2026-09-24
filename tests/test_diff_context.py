import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from agent import (
    CodeReview,
    FindingCategory,
    OverallRating,
    ReviewFinding,
    Severity,
    build_diff_context_node,
    main,
    review_code,
)
from diff_context import DiffContext, DiffLine, parse_unified_diff
from repository_context import RepoContext, RepositoryFile
from static_analysis import StaticAnalysisResult

SINGLE_HUNK_DIFF = """--- a/src/users/service.py
+++ b/src/users/service.py
@@ -10,3 +10,4 @@
 line
-old
+new1
+new2
 line
"""


def successful_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.NEEDS_WORK,
        summary="The supplied change needs review.",
        findings=[
            ReviewFinding(
                category=FindingCategory.BUG,
                severity=Severity.HIGH,
                title="Example regression",
                description="The changed behavior can fail.",
                line_number=1,
                suggestion="Preserve the required behavior.",
            )
        ],
    )


def invalid_review() -> CodeReview:
    return successful_review().model_copy(
        update={"overall_rating": OverallRating.GOOD}
    )


class DiffSchemaTests(unittest.TestCase):
    def test_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            DiffLine.model_validate(
                {
                    "kind": "add",
                    "old_line_number": None,
                    "new_line_number": 1,
                    "content": "value",
                    "unexpected": True,
                }
            )


class UnifiedDiffParserTests(unittest.TestCase):
    def test_single_file_single_hunk_tracks_all_line_kinds(self) -> None:
        context = parse_unified_diff(
            SINGLE_HUNK_DIFF,
            "src/users/service.py",
        )

        self.assertEqual(context.target_file, "src/users/service.py")
        self.assertEqual(len(context.changed_files), 1)
        self.assertEqual(len(context.target_hunks), 1)
        lines = context.target_hunks[0].lines
        self.assertEqual(
            [
                (line.kind, line.old_line_number, line.new_line_number)
                for line in lines
            ],
            [
                ("context", 10, 10),
                ("delete", 11, None),
                ("add", None, 11),
                ("add", None, 12),
                ("context", 12, 13),
            ],
        )
        self.assertEqual(context.changed_new_lines, [11, 12])
        self.assertEqual(
            [
                (line_range.start, line_range.end)
                for line_range in context.changed_new_line_ranges
            ],
            [(11, 12)],
        )

    def test_multiple_hunks_are_preserved_in_input_order(self) -> None:
        diff_text = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
@@ -5 +5,2 @@
 context
+added
"""

        context = parse_unified_diff(diff_text, "app.py")

        self.assertEqual(len(context.target_hunks), 2)
        self.assertEqual(
            [hunk.new_start for hunk in context.target_hunks],
            [1, 5],
        )
        self.assertEqual(context.changed_new_lines, [1, 6])
        self.assertEqual(
            [
                (line_range.start, line_range.end)
                for line_range in context.changed_new_line_ranges
            ],
            [(1, 1), (6, 6)],
        )

    def test_multiple_files_keep_metadata_but_target_only_selects_one(self) -> None:
        diff_text = """--- a/alpha.py
+++ b/alpha.py
@@ -1 +1 @@
-alpha = 1
+alpha = 2
--- a/src/users/service.py
+++ b/src/users/service.py
@@ -3 +3 @@
-return old
+return new
"""

        context = parse_unified_diff(
            diff_text,
            "src/users/service.py",
        )

        self.assertEqual(
            [file.new_path for file in context.changed_files],
            ["alpha.py", "src/users/service.py"],
        )
        self.assertEqual(len(context.target_hunks), 1)
        self.assertEqual(context.target_hunks[0].new_start, 3)
        self.assertEqual(context.changed_new_lines, [3])

    def test_git_a_and_b_prefixes_are_normalized(self) -> None:
        context = parse_unified_diff(
            SINGLE_HUNK_DIFF,
            r"src\users\service.py",
        )

        changed_file = context.changed_files[0]
        self.assertEqual(changed_file.old_path, "src/users/service.py")
        self.assertEqual(changed_file.new_path, "src/users/service.py")
        self.assertEqual(context.target_file, "src/users/service.py")

    def test_new_file_uses_none_for_dev_null_old_path(self) -> None:
        diff_text = """--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+first
+second
"""

        context = parse_unified_diff(diff_text, "new.py")

        changed_file = context.changed_files[0]
        self.assertIsNone(changed_file.old_path)
        self.assertEqual(changed_file.new_path, "new.py")
        self.assertEqual(context.target_hunks[0].old_start, 0)
        self.assertEqual(context.changed_new_lines, [1, 2])

    def test_deleted_file_uses_none_for_dev_null_new_path(self) -> None:
        diff_text = """--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-first
-second
"""

        context = parse_unified_diff(diff_text, "old.py")

        changed_file = context.changed_files[0]
        self.assertEqual(changed_file.old_path, "old.py")
        self.assertIsNone(changed_file.new_path)
        self.assertEqual(context.target_file, "old.py")
        self.assertTrue(
            all(
                line.new_line_number is None
                for line in context.target_hunks[0].lines
            )
        )
        self.assertEqual(context.changed_new_lines, [])

    def test_malformed_hunk_is_bounded_and_warned(self) -> None:
        diff_text = """--- a/app.py
+++ b/app.py
@@ -1,2 +1,1 @@
-only-one-old-line
"""

        context = parse_unified_diff(diff_text, "app.py")

        self.assertEqual(len(context.target_hunks), 1)
        self.assertTrue(
            any("Malformed hunk" in warning for warning in context.warnings)
        )

    def test_malformed_hunk_header_is_warned_without_exception(self) -> None:
        diff_text = """--- a/app.py
+++ b/app.py
@@ malformed @@
+content
"""

        context = parse_unified_diff(diff_text, "app.py")

        self.assertEqual(context.target_hunks, [])
        self.assertTrue(
            any(
                "Malformed hunk header" in warning
                for warning in context.warnings
            )
        )

    def test_target_not_in_diff_has_explicit_warning(self) -> None:
        context = parse_unified_diff(SINGLE_HUNK_DIFF, "src/other.py")

        self.assertEqual(context.target_hunks, [])
        self.assertEqual(context.changed_new_lines, [])
        self.assertIn(
            "Target file was not found in supplied diff.",
            context.warnings,
        )

    def test_empty_diff_returns_empty_context(self) -> None:
        self.assertEqual(
            parse_unified_diff("", "src/users/service.py"),
            DiffContext(),
        )

    def test_diff_char_and_line_budgets_add_warnings(self) -> None:
        with patch("diff_context.MAX_DIFF_CHARS", 25):
            char_bounded = parse_unified_diff(
                SINGLE_HUNK_DIFF,
                "src/users/service.py",
            )
        self.assertTrue(
            any("MAX_DIFF_CHARS=25" in item for item in char_bounded.warnings)
        )

        with patch("diff_context.MAX_DIFF_LINES", 3):
            line_bounded = parse_unified_diff(
                SINGLE_HUNK_DIFF,
                "src/users/service.py",
            )
        self.assertTrue(
            any("MAX_DIFF_LINES=3" in item for item in line_bounded.warnings)
        )

    def test_file_and_target_hunk_budgets_add_warnings(self) -> None:
        multiple_files = """--- a/alpha.py
+++ b/alpha.py
@@ -1 +1 @@
-old
+new
--- a/beta.py
+++ b/beta.py
@@ -1 +1 @@
-old
+new
"""
        with patch("diff_context.MAX_DIFF_FILES", 1):
            file_bounded = parse_unified_diff(multiple_files, "alpha.py")
        self.assertEqual(len(file_bounded.changed_files), 1)
        self.assertTrue(
            any("MAX_DIFF_FILES=1" in item for item in file_bounded.warnings)
        )

        multiple_hunks = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
@@ -3 +3 @@
-old again
+new again
"""
        with patch("diff_context.MAX_TARGET_HUNKS", 1):
            hunk_bounded = parse_unified_diff(multiple_hunks, "app.py")
        self.assertEqual(len(hunk_bounded.target_hunks), 1)
        self.assertTrue(
            any(
                "MAX_TARGET_HUNKS=1" in item
                for item in hunk_bounded.warnings
            )
        )

    def test_repeated_parse_has_deterministic_ordering(self) -> None:
        first = parse_unified_diff(SINGLE_HUNK_DIFF, "src/users/service.py")
        second = parse_unified_diff(SINGLE_HUNK_DIFF, "src/users/service.py")

        self.assertEqual(first, second)
        self.assertEqual(first.changed_new_lines, sorted(first.changed_new_lines))

    def test_unsafe_diff_path_is_not_exposed_as_changed_file(self) -> None:
        diff_text = """--- a/../outside.py
+++ b/../outside.py
@@ -1 +1 @@
-old
+new
"""

        context = parse_unified_diff(diff_text, "outside.py")

        self.assertEqual(context.changed_files, [])
        self.assertTrue(
            any("unsafe" in warning.casefold() for warning in context.warnings)
        )

    def test_simple_rename_is_retained_with_limited_support_warning(self) -> None:
        diff_text = """--- a/old_name.py
+++ b/new_name.py
@@ -1 +1 @@
-old
+new
"""

        context = parse_unified_diff(diff_text, "new_name.py")

        self.assertEqual(context.target_file, "new_name.py")
        self.assertTrue(
            any("Rename has limited support" in item for item in context.warnings)
        )

    def test_binary_diff_content_is_not_parsed(self) -> None:
        diff_text = """--- a/image.png
+++ b/image.png
Binary files a/image.png and b/image.png differ
"""

        context = parse_unified_diff(diff_text, "image.png")

        self.assertTrue(context.changed_files[0].is_binary)
        self.assertEqual(context.target_hunks, [])
        self.assertIn(
            "Binary diff content was not parsed.",
            context.warnings,
        )


class DiffAwareAgentTests(unittest.TestCase):
    @patch("agent.parse_unified_diff")
    def test_no_diff_node_returns_empty_context_without_parsing(
        self,
        diff_parser,
    ) -> None:
        result = build_diff_context_node(
            {
                "diff_text": None,
                "target_file": None,
            }
        )

        self.assertEqual(result, {"diff_context": DiffContext()})
        diff_parser.assert_not_called()

    def test_prompt_contains_target_diff_and_diff_guardrails(self) -> None:
        diff_text = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
"""
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
                "new\n",
                target_file="app.py",
                diff_text=diff_text,
            )

        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("Diff context:", prompt)
        self.assertIn("-old", prompt)
        self.assertIn("+new", prompt)
        self.assertIn("Changed NEW target-file lines: 1", prompt)
        self.assertIn("Prioritize bugs, security regressions", prompt)
        self.assertIn("NEW target-file line numbers", prompt)
        self.assertLess(prompt.index("Review this python code"), prompt.index("Diff context:"))

    def test_no_diff_keeps_full_file_prompt_without_diff_guardrails(self) -> None:
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

            review_code("new\n")

        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("Review this python code", prompt)
        self.assertNotIn("Diff context:", prompt)
        self.assertNotIn("The supplied diff represents", prompt)
        self.assertNotIn("changed area", prompt)

    def test_two_retries_do_not_rebuild_context_diff_or_static_evidence(self) -> None:
        diff_text = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
"""
        repository_context = RepoContext(
            repository_root="repo",
            target_file="app.py",
            related_files=[
                RepositoryFile(
                    path="dependency.py",
                    relationship="local_import",
                    content="CONTRACT = 'repository evidence'\n",
                )
            ],
        )
        analysis = StaticAnalysisResult(
            tool_errors=["synthetic static evidence"]
        )
        with (
            patch(
                "agent.build_repository_context",
                return_value=repository_context,
            ) as repository_builder,
            patch(
                "agent.parse_unified_diff",
                wraps=parse_unified_diff,
            ) as diff_parser,
            patch(
                "agent.analyze_code",
                return_value=analysis,
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
                "new\n",
                repository_root="repo",
                target_file="app.py",
                diff_text=diff_text,
            )

        self.assertEqual(result, successful_review())
        repository_builder.assert_called_once_with("repo", "app.py")
        diff_parser.assert_called_once_with(diff_text, "app.py")
        analyze_code.assert_called_once_with("new\n", "python")
        self.assertEqual(structured_llm.invoke.call_count, 3)

        retry_prompt = structured_llm.invoke.call_args_list[2].args[0][1].content
        self.assertIn("Diff context:", retry_prompt)
        self.assertIn("+new", retry_prompt)
        self.assertIn("CONTRACT = 'repository evidence'", retry_prompt)
        self.assertIn("synthetic static evidence", retry_prompt)
        self.assertIn("previous review failed semantic validation", retry_prompt)
        self.assertIn("semantic retry 2", retry_prompt)

    def test_deleted_guard_and_optional_repository_contract_share_prompt(self) -> None:
        service_source = (
            "from users import repository\n\n"
            "def user_email(user_id: int) -> str:\n"
            "    user = repository.get_user(user_id)\n"
            "    return user.email\n"
        )
        diff_text = """--- a/src/users/service.py
+++ b/src/users/service.py
@@ -1,7 +1,5 @@
 from users import repository
 
 def user_email(user_id: int) -> str:
     user = repository.get_user(user_id)
-    if user is None:
-        raise LookupError("user not found")
     return user.email
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            (repository / "src/users").mkdir(parents=True)
            (repository / "tests/users").mkdir(parents=True)
            (repository / "src/users/__init__.py").write_text(
                "",
                encoding="utf-8",
            )
            (repository / "src/users/service.py").write_text(
                service_source,
                encoding="utf-8",
            )
            (repository / "src/users/repository.py").write_text(
                "def get_user(user_id: int) -> User | None:\n"
                "    ...\n",
                encoding="utf-8",
            )
            (repository / "tests/users/test_service.py").write_text(
                "def test_user_email():\n"
                "    assert True\n",
                encoding="utf-8",
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
                    repository_root=str(repository),
                    target_file="src/users/service.py",
                    diff_text=diff_text,
                )

        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("-    if user is None:", prompt)
        self.assertIn(
            "def get_user(user_id: int) -> User | None:",
            prompt,
        )
        self.assertIn("Relationship: local_import", prompt)
        self.assertLess(
            prompt.index("Diff context:"),
            prompt.index("Repository context:"),
        )

    def test_diff_text_requires_an_explicit_target_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "diff_text requires target_file"):
            review_code("new\n", diff_text=SINGLE_HUNK_DIFF)


class DiffAwareCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name) / "repo"
        self.repository.mkdir()
        self.target = self.repository / "app.py"
        self.target.write_text("new\n", encoding="utf-8")
        self.diff_path = self.repository / "changes.diff"
        self.diff_text = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
"""
        self.diff_path.write_text(self.diff_text, encoding="utf-8")

    @patch("agent.review_code")
    def test_file_with_diff_enables_single_file_diff_mode(
        self,
        mocked_review_code,
    ) -> None:
        mocked_review_code.return_value = successful_review()
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            str(self.target),
            "--diff",
            str(self.diff_path),
        ]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "new\n",
            "python",
            repository_root=None,
            target_file=str(self.target),
            diff_text=self.diff_text,
        )

    @patch("agent.review_code")
    def test_file_repo_and_diff_combine_all_context_modes(
        self,
        mocked_review_code,
    ) -> None:
        mocked_review_code.return_value = successful_review()
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            str(self.repository),
            "--diff",
            str(self.diff_path),
        ]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "new\n",
            "python",
            repository_root=str(self.repository),
            target_file="app.py",
            diff_text=self.diff_text,
        )

    def test_code_with_diff_is_rejected_by_argparse(self) -> None:
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--code",
            "new",
            "--diff",
            str(self.diff_path),
        ]

        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as captured,
        ):
            main()

        self.assertEqual(captured.exception.code, 2)
        self.assertIn("--diff can only be used with --file", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
