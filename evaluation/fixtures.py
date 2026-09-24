"""Generate a small diverse local benchmark of committed Python repositories."""

from __future__ import annotations

import json
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureDefinition:
    task_id: str
    category: str
    task: str
    files: dict[str, str]
    expected_files: tuple[str, ...]


FIXTURES = (
    FixtureDefinition(
        "bugfix-arithmetic",
        "bug_fix",
        "Fix add() so the regression test passes.",
        {
            "src/solution.py": "def add(a, b):\n    return a - b\n",
            "tests/test_solution.py": "from src.solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "edge-clamp",
        "edge_case",
        "Clamp values to the inclusive lower and upper bounds.",
        {
            "src/solution.py": "def clamp(value, low, high):\n    return min(value, high)\n",
            "tests/test_solution.py": "from src.solution import clamp\n\ndef test_bounds():\n    assert clamp(-2, 0, 5) == 0\n    assert clamp(8, 0, 5) == 5\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "feature-even",
        "small_feature",
        "Add an is_even(value) public helper.",
        {
            "src/solution.py": "VALUE = 2\n",
            "tests/test_solution.py": "from src.solution import is_even\n\ndef test_even_and_odd():\n    assert is_even(4) is True\n    assert is_even(3) is False\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "regression-bool",
        "test_regression",
        "Parse false-like strings without treating them as truthy.",
        {
            "src/solution.py": "def parse_bool(value):\n    return bool(value)\n",
            "tests/test_solution.py": "from src.solution import parse_bool\n\ndef test_false_string():\n    assert parse_bool('false') is False\n    assert parse_bool('true') is True\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "refactor-tags",
        "behavioral_refactor",
        "Normalize tags by trimming, deduplicating, and sorting them.",
        {
            "src/solution.py": "def normalize_tags(tags):\n    return list(tags)\n",
            "tests/test_solution.py": "from src.solution import normalize_tags\n\ndef test_normalization():\n    assert normalize_tags([' b ', 'a', 'b']) == ['a', 'b']\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "edge-division",
        "edge_case",
        "Return None instead of raising when safe_divide has a zero denominator.",
        {
            "src/solution.py": "def safe_divide(a, b):\n    return a / b\n",
            "tests/test_solution.py": "from src.solution import safe_divide\n\ndef test_zero():\n    assert safe_divide(5, 0) is None\n    assert safe_divide(6, 2) == 3\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "format-slug",
        "bug_fix",
        "Make slugify collapse repeated whitespace and lowercase words.",
        {
            "src/solution.py": "def slugify(value):\n    return value.replace(' ', '-')\n",
            "tests/test_solution.py": "from src.solution import slugify\n\ndef test_slug():\n    assert slugify('  Hello   World ') == 'hello-world'\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "multi-file-validation",
        "multi_file_fix",
        "Use the shared non-empty validator before saving a display name.",
        {
            "src/validators.py": "def non_empty(value):\n    return bool(value.strip())\n",
            "src/service.py": "def save_name(value, store):\n    store.append(value)\n    return True\n",
            "tests/test_service.py": "from src.service import save_name\n\ndef test_blank_rejected():\n    store = []\n    assert save_name('  ', store) is False\n    assert store == []\n",
        },
        ("src/service.py", "src/validators.py"),
    ),
    FixtureDefinition(
        "cache-key-case",
        "test_regression",
        "Make cache keys case-insensitive while preserving stripped content.",
        {
            "src/solution.py": "def cache_key(value):\n    return value.strip()\n",
            "tests/test_solution.py": "from src.solution import cache_key\n\ndef test_key():\n    assert cache_key(' User@Example.COM ') == 'user@example.com'\n",
        },
        ("src/solution.py",),
    ),
    FixtureDefinition(
        "feature-currency",
        "small_feature",
        "Format integer cents as a signed dollar amount with two decimals.",
        {
            "src/solution.py": "def format_usd(cents):\n    return str(cents)\n",
            "tests/test_solution.py": "from src.solution import format_usd\n\ndef test_currency():\n    assert format_usd(1234) == '$12.34'\n    assert format_usd(-5) == '-$0.05'\n",
        },
        ("src/solution.py",),
    ),
)


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def create_local_benchmark(destination: str | Path) -> Path:
    """Create ten independent committed repos and their trusted JSONL metadata."""

    root = Path(destination).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Fixture destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    sources = root / "sources"
    sources.mkdir()
    records: list[dict[str, object]] = []
    for definition in FIXTURES:
        repository = sources / definition.task_id
        repository.mkdir()
        for relative, content in definition.files.items():
            target = repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (repository / "src" / "__init__.py").write_text("", encoding="utf-8")
        _git(repository, "init", "-b", "main")
        _git(repository, "config", "user.name", "RepoGraph Evaluation Fixtures")
        _git(repository, "config", "user.email", "evaluation@example.invalid")
        _git(repository, "config", "core.autocrlf", "false")
        _git(repository, "add", ".")
        _git(repository, "commit", "-m", "benchmark base")
        records.append(
            {
                "id": definition.task_id,
                "dataset": "repograph-local-v1",
                "repository": str(repository),
                "base_commit": _git(repository, "rev-parse", "HEAD"),
                "task": definition.task,
                "test_command": ["python", "-m", "pytest", "tests", "-q"],
                "expected_files": list(definition.expected_files),
                "metadata": {"category": definition.category, "fixture_version": 1},
            }
        )
    dataset = root / "local-v1.jsonl"
    dataset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    return dataset
