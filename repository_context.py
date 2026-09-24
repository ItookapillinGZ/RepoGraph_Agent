"""Deterministic, bounded repository context for local Python reviews."""

import ast
import os
import tokenize
from bisect import insort
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXCLUDED_DIRECTORIES = {
    ".eggs",
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "site-packages",
    "venv",
}
REPOSITORY_FILE_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
MAX_TREE_ENTRIES = 200
MAX_RELATED_FILES = 8
MAX_RELATED_TEST_FILES = 3
MAX_FILE_CHARS = 12_000
MAX_TOTAL_CONTEXT_CHARS = 40_000
MAX_TARGET_FILE_CHARS = 200_000

Relationship = Literal["local_import", "test"]


class RepositoryFile(BaseModel):
    """One bounded repository file supplied as review context."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    relationship: Relationship
    content: str


class RepoContext(BaseModel):
    """Deterministic repository context for one target file."""

    model_config = ConfigDict(extra="forbid")

    repository_root: str | None = None
    target_file: str | None = None
    file_tree: list[str] = Field(default_factory=list)
    related_files: list[RepositoryFile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True, order=True)
class PythonImport:
    """A normalized Python import extracted without importing user code."""

    level: int
    module: str
    names: tuple[str, ...] = ()


def extract_python_imports(source: str) -> list[PythonImport]:
    """Parse and normalize imports from Python source without executing it."""

    tree = ast.parse(source)
    imports: set[PythonImport] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                PythonImport(level=0, module=alias.name)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            imports.add(
                PythonImport(
                    level=node.level,
                    module=node.module or "",
                    names=tuple(sorted(alias.name for alias in node.names)),
                )
            )

    return sorted(imports)


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _is_within(path: Path, repository_root: Path) -> bool:
    return path == repository_root or repository_root in path.parents


def _validated_paths(
    repository_root: str,
    target_file: str,
) -> tuple[Path, Path]:
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Repository root cannot be resolved: {repository_root}") from error

    if not root.is_dir():
        raise ValueError(f"Repository root is not a directory: {repository_root}")

    raw_target = Path(target_file)
    target_candidate = raw_target if raw_target.is_absolute() else root / raw_target
    try:
        target = target_candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Target file cannot be resolved: {target_file}") from error

    if not _is_within(target, root):
        raise ValueError("Target file must resolve inside repository root.")
    if not target.is_file():
        raise ValueError(f"Target path is not a file: {target_file}")
    if target.suffix.casefold() != ".py":
        raise ValueError("Repository-aware context currently requires a Python target file.")

    return root, target


def _safe_repository_file(path: Path, repository_root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or not _is_within(resolved, repository_root):
        return None
    return resolved


def _walk_files(repository_root: Path):
    excluded = {name.casefold() for name in EXCLUDED_DIRECTORIES}

    for current, directory_names, file_names in os.walk(
        repository_root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        directory_names[:] = [
            name
            for name in sorted(directory_names)
            if name.casefold() not in excluded
            and not (current_path / name).is_symlink()
        ]
        for file_name in sorted(file_names):
            yield current_path / file_name


def _discover_file_tree(
    repository_root: Path,
    warnings: list[str],
) -> list[str]:
    entries: list[str] = []
    truncated = False
    skipped_unsafe_file = False

    for candidate in _walk_files(repository_root):
        if candidate.suffix.casefold() not in REPOSITORY_FILE_SUFFIXES:
            continue
        resolved = _safe_repository_file(candidate, repository_root)
        if resolved is None:
            skipped_unsafe_file = True
            continue
        relative_path = resolved.relative_to(repository_root).as_posix()
        if relative_path in entries:
            continue
        insort(entries, relative_path)
        if len(entries) > MAX_TREE_ENTRIES:
            entries.pop()
            truncated = True

    if truncated:
        _add_warning(
            warnings,
            f"File tree truncated at MAX_TREE_ENTRIES={MAX_TREE_ENTRIES}.",
        )
    if skipped_unsafe_file:
        _add_warning(
            warnings,
            "Skipped one or more files that resolve outside repository root.",
        )
    return entries


def _module_candidates(base: Path, parts: tuple[str, ...]) -> tuple[Path, Path]:
    module_path = base.joinpath(*parts)
    return module_path.with_suffix(".py"), module_path / "__init__.py"


def _import_candidate_paths(
    imported: PythonImport,
    repository_root: Path,
    target_file: Path,
) -> list[Path]:
    module_parts = tuple(part for part in imported.module.split(".") if part)
    candidates: list[Path] = []

    if imported.level:
        base = target_file.parent
        for _ in range(imported.level - 1):
            base = base.parent
        bases = [base]
    else:
        bases = [repository_root, repository_root / "src"]

    for base in bases:
        if module_parts:
            candidates.extend(_module_candidates(base, module_parts))
        for imported_name in imported.names:
            if imported_name == "*":
                continue
            candidates.extend(
                _module_candidates(base, (*module_parts, imported_name))
            )

    return candidates


def _resolve_local_imports(
    imports: list[PythonImport],
    repository_root: Path,
    target_file: Path,
    warnings: list[str],
) -> list[Path]:
    resolved_imports: dict[str, Path] = {}
    skipped_unsafe_file = False

    for imported in imports:
        for candidate in _import_candidate_paths(
            imported,
            repository_root,
            target_file,
        ):
            resolved = _safe_repository_file(candidate, repository_root)
            if resolved is None:
                if candidate.exists():
                    skipped_unsafe_file = True
                continue
            if resolved == target_file or resolved.suffix.casefold() != ".py":
                continue
            relative_path = resolved.relative_to(repository_root).as_posix()
            resolved_imports[relative_path] = resolved

    if skipped_unsafe_file:
        _add_warning(
            warnings,
            "Skipped one or more imported files that resolve outside repository root.",
        )
    return [resolved_imports[path] for path in sorted(resolved_imports)]


def _related_test_candidates(
    repository_root: Path,
    target_file: Path,
    warnings: list[str],
) -> list[Path]:
    target_relative = target_file.relative_to(repository_root)
    relative_parent_parts = target_relative.parent.parts
    if relative_parent_parts[:1] == ("src",):
        relative_parent_parts = relative_parent_parts[1:]

    test_names = (f"test_{target_file.stem}.py", f"{target_file.stem}_test.py")
    tests_root = repository_root / "tests"
    priority_bases = [tests_root]
    if relative_parent_parts:
        priority_bases.append(tests_root.joinpath(*relative_parent_parts))
    priority_bases.append(target_file.parent)

    ordered: list[Path] = []
    seen: set[str] = set()

    def add_candidate(candidate: Path) -> None:
        resolved = _safe_repository_file(candidate, repository_root)
        if resolved is None or resolved == target_file:
            return
        relative_path = resolved.relative_to(repository_root).as_posix()
        if relative_path in seen:
            return
        seen.add(relative_path)
        ordered.append(resolved)

    for test_name in test_names:
        for base in priority_bases:
            add_candidate(base / test_name)

    recursive_matches: list[tuple[str, Path]] = []
    if tests_root.is_dir() and not tests_root.is_symlink():
        for candidate in _walk_files(tests_root):
            if candidate.name not in test_names:
                continue
            resolved = _safe_repository_file(candidate, repository_root)
            if resolved is None:
                continue
            relative_path = resolved.relative_to(repository_root).as_posix()
            if relative_path in seen:
                continue
            insort(recursive_matches, (relative_path, resolved))
            if len(recursive_matches) > MAX_RELATED_TEST_FILES + 1:
                recursive_matches.pop()

    for _, candidate in recursive_matches:
        add_candidate(candidate)

    if len(ordered) > MAX_RELATED_TEST_FILES:
        _add_warning(
            warnings,
            "Related test discovery truncated at "
            f"MAX_RELATED_TEST_FILES={MAX_RELATED_TEST_FILES}.",
        )
    return ordered[:MAX_RELATED_TEST_FILES]


def _read_limited_python(path: Path, limit: int) -> tuple[str, bool]:
    with tokenize.open(path) as source_file:
        content = source_file.read(limit + 1)
    return content[:limit], len(content) > limit


def _target_imports(target_file: Path, warnings: list[str]) -> list[PythonImport]:
    source, truncated = _read_limited_python(target_file, MAX_TARGET_FILE_CHARS)
    if truncated:
        _add_warning(
            warnings,
            "Target import extraction skipped because target exceeds "
            f"MAX_TARGET_FILE_CHARS={MAX_TARGET_FILE_CHARS}.",
        )
        return []
    try:
        return extract_python_imports(source)
    except SyntaxError as error:
        _add_warning(
            warnings,
            f"Target imports could not be parsed: {error.msg}.",
        )
        return []


def read_repository_target(repository_root: str, target_file: str) -> str:
    """Read a repository target only after containment and symlink validation."""

    _, target = _validated_paths(repository_root, target_file)
    source, truncated = _read_limited_python(target, MAX_TARGET_FILE_CHARS)
    if truncated:
        raise ValueError(
            "Target file exceeds "
            f"MAX_TARGET_FILE_CHARS={MAX_TARGET_FILE_CHARS}."
        )
    return source


def _bounded_related_files(
    candidates: list[tuple[Path, Relationship]],
    repository_root: Path,
    warnings: list[str],
) -> list[RepositoryFile]:
    unique_candidates: list[tuple[Path, Relationship]] = []
    seen: set[str] = set()
    for path, relationship in candidates:
        relative_path = path.relative_to(repository_root).as_posix()
        if relative_path in seen:
            continue
        seen.add(relative_path)
        unique_candidates.append((path, relationship))

    if len(unique_candidates) > MAX_RELATED_FILES:
        _add_warning(
            warnings,
            f"Related files truncated at MAX_RELATED_FILES={MAX_RELATED_FILES}.",
        )
    unique_candidates = unique_candidates[:MAX_RELATED_FILES]

    related_files: list[RepositoryFile] = []
    total_chars = 0
    for path, relationship in unique_candidates:
        remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            _add_warning(
                warnings,
                "Related context truncated because MAX_TOTAL_CONTEXT_CHARS was reached.",
            )
            break

        file_limit = min(MAX_FILE_CHARS, remaining)
        relative_path = path.relative_to(repository_root).as_posix()
        try:
            content, truncated = _read_limited_python(path, file_limit)
        except (OSError, SyntaxError, UnicodeError) as error:
            _add_warning(
                warnings,
                f"Related file could not be read: {relative_path}: {error}.",
            )
            continue

        related_files.append(
            RepositoryFile(
                path=relative_path,
                relationship=relationship,
                content=content,
            )
        )
        total_chars += len(content)
        if truncated:
            if file_limit < MAX_FILE_CHARS:
                _add_warning(
                    warnings,
                    "Related context truncated because "
                    "MAX_TOTAL_CONTEXT_CHARS was reached.",
                )
                break
            _add_warning(
                warnings,
                "Related file content truncated at "
                f"MAX_FILE_CHARS={MAX_FILE_CHARS}: {relative_path}.",
            )

    return related_files


def build_repository_context(
    repository_root: str,
    target_file: str,
) -> RepoContext:
    """Build safe, deterministic, and bounded context for a Python target."""

    root, target = _validated_paths(repository_root, target_file)
    warnings: list[str] = []
    file_tree = _discover_file_tree(root, warnings)
    imports = _target_imports(target, warnings)
    local_imports = _resolve_local_imports(imports, root, target, warnings)
    related_tests = _related_test_candidates(root, target, warnings)
    candidates: list[tuple[Path, Relationship]] = [
        (path, "local_import") for path in local_imports
    ]
    candidates.extend((path, "test") for path in related_tests)
    related_files = _bounded_related_files(candidates, root, warnings)

    return RepoContext(
        repository_root=str(root),
        target_file=target.relative_to(root).as_posix(),
        file_tree=file_tree,
        related_files=related_files,
        warnings=warnings,
    )
