"""Tests for the root-bound, read-only repository tools."""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from repository_tools import (
    MAX_LIST_RESULTS,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_FILE_CHARS,
    UNTRUSTED_CONTENT_HEADER,
    build_repository_tools,
)


class RepositoryToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "service.py").write_text(
            "from .repository import find_by_email\n"
            "user = find_by_email(email)\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "def find_by_email(email):\n"
            "    return USERS.get(email)\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_find_by_email():\n"
            "    assert find_by_email('a@example.com')\n",
            encoding="utf-8",
        )
        self.tools = {
            tool.name: tool for tool in build_repository_tools(str(self.root))
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_factory_validates_root_and_exposes_exactly_three_tools(self) -> None:
        self.assertEqual(
            set(self.tools),
            {
                "read_repository_file",
                "search_repository_code",
                "list_repository_files",
            },
        )
        with self.assertRaises(ValueError):
            build_repository_tools(str(self.root / "missing"))
        file_path = self.root / "not-a-directory"
        file_path.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_repository_tools(str(file_path))

    def test_tool_schemas_never_expose_repository_root(self) -> None:
        for tool in self.tools.values():
            self.assertNotIn("repository_root", tool.args)
        self.assertEqual(set(self.tools["read_repository_file"].args), {"path"})
        self.assertEqual(
            set(self.tools["search_repository_code"].args),
            {"query", "max_results"},
        )
        self.assertEqual(
            set(self.tools["list_repository_files"].args),
            {"prefix", "max_results"},
        )

    def test_valid_python_file_is_read_and_marked_untrusted(self) -> None:
        output = self.tools["read_repository_file"].invoke(
            {"path": "src/repository.py"}
        )
        self.assertIn(UNTRUSTED_CONTENT_HEADER, output)
        self.assertIn("Path: src/repository.py", output)
        self.assertIn("def find_by_email", output)
        self.assertNotIn(str(self.root), output)

    def test_read_rejects_absolute_traversal_and_escape_paths(self) -> None:
        tool = self.tools["read_repository_file"]
        for path in (
            "../outside.py",
            "src/../../outside.py",
            str((self.root / "src" / "service.py").resolve()),
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                tool.invoke({"path": path})

    def test_read_rejects_excluded_and_secret_paths(self) -> None:
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config.py").write_text("TOKEN = 1")
        (self.root / ".venv").mkdir()
        (self.root / ".venv" / "secret.py").write_text("TOKEN = 1")
        for path in (".git/config.py", ".venv/secret.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tools["read_repository_file"].invoke({"path": path})

        for name in (".env", ".env.local", "private.pem", "private.key"):
            (self.root / name).write_text("SECRET=value", encoding="utf-8")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.tools["read_repository_file"].invoke({"path": name})

    def test_read_rejects_unsupported_and_binary_files(self) -> None:
        (self.root / "image.bin").write_bytes(b"plain")
        (self.root / "src" / "binary.py").write_bytes(b"value = 1\x00binary")
        with self.assertRaises(ValueError):
            self.tools["read_repository_file"].invoke({"path": "image.bin"})
        with self.assertRaises(ValueError):
            self.tools["read_repository_file"].invoke({"path": "src/binary.py"})

    def test_read_rejects_file_and_directory_symlinks(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside.py"
        outside.write_text("SECRET = 1", encoding="utf-8")
        file_link = self.root / "src" / "link.py"
        directory_link = self.root / "linked"
        try:
            file_link.symlink_to(outside)
            directory_link.symlink_to(self.root / "src", target_is_directory=True)
        except OSError as error:
            outside.unlink(missing_ok=True)
            self.skipTest(f"Symlinks unavailable: {error}")
        try:
            with self.assertRaises(ValueError):
                self.tools["read_repository_file"].invoke({"path": "src/link.py"})
            with self.assertRaises(ValueError):
                self.tools["read_repository_file"].invoke(
                    {"path": "linked/service.py"}
                )
        finally:
            file_link.unlink(missing_ok=True)
            directory_link.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)

    def test_file_read_is_truncated_with_an_explicit_marker(self) -> None:
        content = "x" * (MAX_TOOL_FILE_CHARS + 1)
        (self.root / "large.md").write_text(content, encoding="utf-8")
        output = self.tools["read_repository_file"].invoke({"path": "large.md"})
        self.assertIn(
            f"[TRUNCATED at {MAX_TOOL_FILE_CHARS} characters]",
            output,
        )
        self.assertNotIn("x" * (MAX_TOOL_FILE_CHARS + 1), output)

    def test_literal_search_is_deterministic_and_marked_untrusted(self) -> None:
        output = self.tools["search_repository_code"].invoke(
            {"query": "find_by_email"}
        )
        self.assertIn(UNTRUSTED_CONTENT_HEADER, output)
        service_index = output.index("src/service.py:1:")
        repository_index = output.index("src/repository.py:1:")
        test_index = output.index("tests/test_service.py:1:")
        self.assertLess(repository_index, service_index)
        self.assertLess(service_index, test_index)

    def test_search_uses_literal_not_regex_semantics(self) -> None:
        output = self.tools["search_repository_code"].invoke({"query": ".*"})
        self.assertIn("No literal matches found.", output)

    def test_search_reads_beyond_the_single_file_read_budget(self) -> None:
        content = ("padding = 1\n" * 1200) + "late_needle = True\n"
        self.assertGreater(len(content), MAX_TOOL_FILE_CHARS)
        (self.root / "src" / "late.py").write_text(content, encoding="utf-8")
        output = self.tools["search_repository_code"].invoke(
            {"query": "late_needle"}
        )
        self.assertIn("src/late.py:1201:", output)

    def test_search_rejects_empty_and_overlong_queries(self) -> None:
        tool = self.tools["search_repository_code"]
        with self.assertRaises(ValidationError):
            tool.invoke({"query": ""})
        with self.assertRaises(ValidationError):
            tool.invoke({"query": "x" * 201})

    def test_search_result_count_and_line_length_are_bounded(self) -> None:
        (self.root / "src" / "many.py").write_text(
            "\n".join(f"MATCH_{index} = 'needle'" for index in range(25)),
            encoding="utf-8",
        )
        output = self.tools["search_repository_code"].invoke(
            {"query": "needle", "max_results": 2}
        )
        self.assertEqual(output.count("src/many.py:"), 2)
        self.assertIn("[TRUNCATED at 2 search results]", output)
        with self.assertRaises(ValidationError):
            self.tools["search_repository_code"].invoke(
                {"query": "needle", "max_results": MAX_SEARCH_RESULTS + 1}
            )

    def test_list_root_and_prefix_are_deterministic_and_relative(self) -> None:
        root_output = self.tools["list_repository_files"].invoke({})
        listed = [
            line
            for line in root_output.splitlines()
            if line.endswith((".py", ".md"))
        ]
        self.assertEqual(listed, sorted(listed))
        self.assertNotIn(str(self.root), root_output)
        prefix_output = self.tools["list_repository_files"].invoke(
            {"prefix": "src"}
        )
        self.assertIn("src/repository.py", prefix_output)
        self.assertNotIn("tests/test_service.py", prefix_output)

    def test_list_excludes_secret_and_excluded_directories(self) -> None:
        (self.root / ".env").write_text("SECRET=1", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "hidden.py").write_text("x = 1")
        output = self.tools["list_repository_files"].invoke({})
        self.assertNotIn(".env", output)
        self.assertNotIn(".git/hidden.py", output)

    def test_list_result_count_is_bounded(self) -> None:
        for index in range(5):
            (self.root / "src" / f"file_{index}.py").write_text("x = 1")
        output = self.tools["list_repository_files"].invoke(
            {"prefix": "src", "max_results": 2}
        )
        result_lines = [
            line for line in output.splitlines() if line.startswith("src/")
        ]
        self.assertEqual(len(result_lines), 2)
        self.assertIn("[TRUNCATED at 2 file results]", output)
        with self.assertRaises(ValidationError):
            self.tools["list_repository_files"].invoke(
                {"max_results": MAX_LIST_RESULTS + 1}
            )

    def test_all_tools_forbid_extra_arguments(self) -> None:
        with self.assertRaises(ValidationError):
            self.tools["read_repository_file"].invoke(
                {"path": "src/service.py", "repository_root": str(self.root)}
            )


if __name__ == "__main__":
    unittest.main()
