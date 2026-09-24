import unittest

from pydantic import ValidationError

from fix_context import (
    CodeFix,
    build_fix_patch,
    normalize_target_path,
    validate_candidate_fix,
)


class CodeFixSchemaTests(unittest.TestCase):
    def test_schema_accepts_complete_updated_source(self) -> None:
        fix = CodeFix(
            summary="Correct the operation.",
            addressed_findings=["Wrong arithmetic operation"],
            updated_code="def add(a, b):\n    return a + b\n",
        )
        self.assertIn("return a + b", fix.updated_code)

    def test_schema_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            CodeFix.model_validate(
                {
                    "summary": "Fix it.",
                    "addressed_findings": [],
                    "updated_code": "VALUE = 1\n",
                    "repository_path": "app.py",
                }
            )

    def test_schema_rejects_empty_summary(self) -> None:
        with self.assertRaises(ValidationError):
            CodeFix(summary="", addressed_findings=[], updated_code="VALUE = 1\n")

    def test_schema_rejects_empty_updated_code(self) -> None:
        with self.assertRaises(ValidationError):
            CodeFix(summary="Fix it.", addressed_findings=[], updated_code="")


class CandidateValidationTests(unittest.TestCase):
    original = "def add(a, b):\n    return a - b\n"
    title = "Wrong arithmetic operation"

    def fix(self, updated_code: str, addressed: list[str] | None = None) -> CodeFix:
        return CodeFix(
            summary="Correct the operation.",
            addressed_findings=[self.title] if addressed is None else addressed,
            updated_code=updated_code,
        )

    def validate(self, fix: CodeFix, **options):
        return validate_candidate_fix(
            fix,
            self.original,
            [self.title],
            "src/calculator.py",
            **options,
        )

    def test_valid_candidate_has_non_empty_patch(self) -> None:
        result = self.validate(
            self.fix("def add(a, b):\n    return a + b\n")
        )
        self.assertTrue(result.is_valid)
        self.assertTrue(result.patch)

    def test_missing_candidate_is_invalid(self) -> None:
        result = validate_candidate_fix(
            None,
            self.original,
            [self.title],
            "src/calculator.py",
        )
        self.assertIn("No candidate fix", result.errors[0])

    def test_unchanged_source_is_invalid(self) -> None:
        result = self.validate(self.fix(self.original))
        self.assertTrue(any("unchanged" in error for error in result.errors))
        self.assertTrue(any("patch must not be empty" in error for error in result.errors))

    def test_whitespace_only_source_is_invalid(self) -> None:
        result = self.validate(self.fix("   \n"))
        self.assertTrue(any("empty or whitespace" in error for error in result.errors))

    def test_invalid_python_syntax_is_structured_feedback(self) -> None:
        result = self.validate(self.fix("def broken(:\n    pass\n"))
        self.assertTrue(any("SyntaxError" in error for error in result.errors))

    def test_markdown_fenced_source_is_rejected(self) -> None:
        result = self.validate(self.fix("```python\nVALUE = 1\n```"))
        self.assertTrue(any("Markdown code fences" in error for error in result.errors))

    def test_addressed_finding_must_exist_in_review(self) -> None:
        result = self.validate(
            self.fix("VALUE = 1\n", addressed=["Invented finding"])
        )
        self.assertTrue(any("not present in the review" in error for error in result.errors))

    def test_duplicate_addressed_titles_are_rejected(self) -> None:
        result = self.validate(
            self.fix(
                "VALUE = 1\n",
                addressed=[self.title, f"  {self.title.upper()}  "],
            )
        )
        self.assertTrue(any("Duplicate addressed" in error for error in result.errors))

    def test_updated_source_size_is_bounded(self) -> None:
        result = self.validate(self.fix("x" * 11), max_code_chars=10)
        self.assertTrue(any("MAX_FIXED_CODE_CHARS=10" in error for error in result.errors))


class DeterministicPatchTests(unittest.TestCase):
    def test_unified_diff_has_repository_relative_a_and_b_paths(self) -> None:
        patch = build_fix_patch("VALUE = 1\n", "VALUE = 2\n", "src/app.py")
        self.assertEqual(
            patch,
            "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
        )

    def test_repeated_generation_is_identical(self) -> None:
        first = build_fix_patch("a\n", "b\n", "app.py")
        second = build_fix_patch("a\n", "b\n", "app.py")
        self.assertEqual(first, second)

    def test_windows_separators_are_normalized(self) -> None:
        patch = build_fix_patch("a\n", "b\n", "src\\app.py")
        self.assertIn("--- a/src/app.py", patch)
        self.assertEqual(normalize_target_path("src\\app.py"), "src/app.py")

    def test_parent_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            normalize_target_path("../outside.py")

    def test_absolute_windows_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            normalize_target_path("C:\\outside.py")

    def test_unchanged_source_has_no_patch(self) -> None:
        self.assertEqual(build_fix_patch("VALUE = 1\n", "VALUE = 1\n", "app.py"), "")


if __name__ == "__main__":
    unittest.main()
