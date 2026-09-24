from __future__ import annotations

import os
import unittest
from pathlib import Path

from observability.cli import build_parser as build_observability_parser
from plan_execution import validate_multi_file_candidate
from repograph.__main__ import build_parser
from repograph.demo import _candidate, _fixture_identity, _plan
from repograph.doctor import collect_checks

ROOT = Path(__file__).resolve().parent.parent


class H4ProductizationTests(unittest.TestCase):
    def test_unified_entrypoints_parse(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["doctor"]).command, "doctor")
        self.assertEqual(parser.parse_args(["demo"]).sandbox, "docker")
        self.assertEqual(
            parser.parse_args(["release-check"]).command, "release-check"
        )
        self.assertEqual(parser.parse_args(["hygiene"]).command, "hygiene")

    def test_documented_observability_commands_parse(self) -> None:
        parser = build_observability_parser()
        for argv in (
            ["runs"], ["show", "run-id"], ["artifacts", "run-id"],
            ["lineage", "run-id"], ["export", "run-id"],
            ["replay", "run-id", "--candidate", "artifact-id"],
        ):
            with self.subTest(argv=argv):
                self.assertIsNotNone(parser.parse_args(argv))

    def test_demo_candidate_matches_real_core_validation(self) -> None:
        errors = validate_multi_file_candidate(
            _candidate(), _plan(), str(ROOT / "demo" / "fixture_repo")
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(_fixture_identity()), 64)

    def test_doctor_never_contains_secret_values(self) -> None:
        checks = collect_checks()
        rendered = "\n".join(
            f"{item.name} {item.status} {item.detail}" for item in checks
        )
        for name in (
            "CODE_REVIEW_LLM_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"
        ):
            value = os.environ.get(name)
            if value:
                self.assertNotIn(value, rendered)

    def test_public_entrypoint_files_exist(self) -> None:
        paths = (
            "scripts/doctor.ps1", "scripts/demo.ps1", "scripts/start-studio.ps1",
            "scripts/release-check.ps1", "docs/ARCHITECTURE.md",
            "docs/EVALUATION.md", "docs/DEMO.md", "docs/PORTFOLIO.md",
            "docs/PROJECT_INVENTORY.md", "SECURITY.md",
        )
        self.assertEqual([path for path in paths if not (ROOT / path).is_file()], [])

    def test_public_docs_have_no_developer_absolute_path(self) -> None:
        public = [ROOT / "README.md", ROOT / "SECURITY.md", ROOT / ".env.example"]
        public.extend((ROOT / "docs").glob("*.md"))
        public.extend((ROOT / "scripts").glob("*.ps1"))
        # Secret-stripped acceptance copies intentionally omit .env.example.
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in public if path.is_file()
        )
        self.assertNotIn("C:\\Users\\", combined)
        self.assertNotIn("C:/Users/", combined)


if __name__ == "__main__":
    unittest.main()
