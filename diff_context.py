"""Deterministic, bounded parsing for user-supplied unified diffs."""

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_DIFF_CHARS = 50_000
MAX_DIFF_FILES = 20
MAX_TARGET_HUNKS = 20
MAX_DIFF_LINES = 1_000

_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:/")


class DiffLine(BaseModel):
    """One context, addition, or deletion line from a unified-diff hunk."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["context", "add", "delete"]
    old_line_number: int | None = Field(default=None, ge=1)
    new_line_number: int | None = Field(default=None, ge=1)
    content: str


class DiffHunk(BaseModel):
    """One parsed unified-diff hunk with old and new line coordinates."""

    model_config = ConfigDict(extra="forbid")

    old_start: int = Field(ge=0)
    old_count: int = Field(ge=0)
    new_start: int = Field(ge=0)
    new_count: int = Field(ge=0)
    lines: list[DiffLine] = Field(default_factory=list)


class ChangedFile(BaseModel):
    """Bounded metadata and parsed hunks for one changed file."""

    model_config = ConfigDict(extra="forbid")

    old_path: str | None = None
    new_path: str | None = None
    hunks: list[DiffHunk] = Field(default_factory=list)
    is_binary: bool = False


class ChangedLineRange(BaseModel):
    """One inclusive range of changed lines in the new target file."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=1)
    end: int = Field(ge=1)


class DiffContext(BaseModel):
    """Structured change context for one review target."""

    model_config = ConfigDict(extra="forbid")

    target_file: str | None = None
    changed_files: list[ChangedFile] = Field(default_factory=list)
    target_hunks: list[DiffHunk] = Field(default_factory=list)
    changed_new_lines: list[int] = Field(default_factory=list)
    changed_new_line_ranges: list[ChangedLineRange] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)


@dataclass
class _HunkBuilder:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    input_line_number: int
    old_line_number: int = field(init=False)
    new_line_number: int = field(init=False)
    old_seen: int = 0
    new_seen: int = 0
    lines: list[DiffLine] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.old_line_number = self.old_start
        self.new_line_number = self.new_start

    @property
    def complete(self) -> bool:
        return (
            self.old_seen == self.old_count
            and self.new_seen == self.new_count
        )

    def add_line(self, raw_line: str) -> bool:
        if not raw_line or raw_line[0] not in {" ", "+", "-"}:
            return False

        prefix = raw_line[0]
        content = raw_line[1:]
        if prefix == " ":
            self.lines.append(
                DiffLine(
                    kind="context",
                    old_line_number=self.old_line_number,
                    new_line_number=self.new_line_number,
                    content=content,
                )
            )
            self.old_line_number += 1
            self.new_line_number += 1
            self.old_seen += 1
            self.new_seen += 1
        elif prefix == "+":
            self.lines.append(
                DiffLine(
                    kind="add",
                    new_line_number=self.new_line_number,
                    content=content,
                )
            )
            self.new_line_number += 1
            self.new_seen += 1
        else:
            self.lines.append(
                DiffLine(
                    kind="delete",
                    old_line_number=self.old_line_number,
                    content=content,
                )
            )
            self.old_line_number += 1
            self.old_seen += 1
        return True

    def build(self) -> DiffHunk:
        return DiffHunk(
            old_start=self.old_start,
            old_count=self.old_count,
            new_start=self.new_start,
            new_count=self.new_count,
            lines=self.lines,
        )


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _normalize_header_path(
    header_line: str,
    warnings: list[str],
    input_line_number: int,
) -> tuple[str | None, bool]:
    raw_path = header_line[4:].split("\t", 1)[0].strip()
    if raw_path == "/dev/null":
        return None, True

    path = raw_path.replace("\\", "/")
    if path.startswith(("a/", "b/")):
        path = path[2:]

    pure_path = PurePosixPath(path)
    if (
        not path
        or path.startswith("/")
        or _WINDOWS_ABSOLUTE_PATH.match(path)
        or ".." in pure_path.parts
        or pure_path.as_posix() == "."
    ):
        _add_warning(
            warnings,
            "Ignored unsafe or non-repository-relative diff path at "
            f"input line {input_line_number}: {raw_path or '<empty>'}.",
        )
        return None, False

    return pure_path.as_posix(), True


def _bounded_input(diff_text: str, warnings: list[str]) -> list[str]:
    bounded_text = diff_text
    if len(bounded_text) > MAX_DIFF_CHARS:
        bounded_text = bounded_text[:MAX_DIFF_CHARS]
        _add_warning(
            warnings,
            f"Diff text truncated at MAX_DIFF_CHARS={MAX_DIFF_CHARS}.",
        )

    lines = bounded_text.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES]
        _add_warning(
            warnings,
            f"Diff text truncated at MAX_DIFF_LINES={MAX_DIFF_LINES}.",
        )
    return lines


def _parse_hunk_header(
    line: str,
    input_line_number: int,
) -> _HunkBuilder | None:
    match = _HUNK_HEADER.match(line)
    if match is None:
        return None

    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or "1")
    return _HunkBuilder(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        input_line_number=input_line_number,
    )


def _line_ranges(line_numbers: list[int]) -> list[ChangedLineRange]:
    if not line_numbers:
        return []

    ranges: list[ChangedLineRange] = []
    range_start = line_numbers[0]
    range_end = line_numbers[0]
    for line_number in line_numbers[1:]:
        if line_number == range_end + 1:
            range_end = line_number
            continue
        ranges.append(ChangedLineRange(start=range_start, end=range_end))
        range_start = line_number
        range_end = line_number
    ranges.append(ChangedLineRange(start=range_start, end=range_end))
    return ranges


def _target_match(
    changed_files: list[ChangedFile],
    target_file: str,
) -> int | None:
    target = target_file.replace("\\", "/")
    if target.startswith(("a/", "b/")):
        target = target[2:]
    target = target.rstrip("/")

    exact_matches: set[int] = set()
    suffix_matches: set[int] = set()
    for index, changed_file in enumerate(changed_files):
        for path in (changed_file.old_path, changed_file.new_path):
            if path is None:
                continue
            if target == path:
                exact_matches.add(index)
            elif target.endswith(f"/{path}") or path.endswith(f"/{target}"):
                suffix_matches.add(index)

    if len(exact_matches) == 1:
        return next(iter(exact_matches))
    if len(exact_matches) > 1:
        return None
    if len(suffix_matches) == 1:
        return next(iter(suffix_matches))
    return None


def _fallback_target_path(target_file: str) -> str | None:
    target = target_file.replace("\\", "/").rstrip("/")
    if target.startswith(("a/", "b/")):
        target = target[2:]
    if not target:
        return None

    pure_path = PurePosixPath(target)
    if (
        target.startswith("/")
        or _WINDOWS_ABSOLUTE_PATH.match(target)
        or ".." in pure_path.parts
    ):
        return pure_path.name or None
    return pure_path.as_posix()


def parse_unified_diff(
    diff_text: str | None,
    target_file: str | None = None,
) -> DiffContext:
    """Parse bounded unified-diff text without reading files or running Git."""

    if not diff_text:
        return DiffContext()

    warnings: list[str] = []
    input_lines = _bounded_input(diff_text, warnings)
    changed_files: list[ChangedFile] = []
    current_file: ChangedFile | None = None
    current_hunk: _HunkBuilder | None = None
    pending_old_path: str | None = None
    pending_old_path_valid = False
    pending_old_header = False

    def finish_hunk() -> None:
        nonlocal current_hunk
        if current_hunk is None:
            return
        if not current_hunk.complete:
            _add_warning(
                warnings,
                "Malformed hunk at input line "
                f"{current_hunk.input_line_number}: expected "
                f"{current_hunk.old_count} old/{current_hunk.new_count} new "
                f"lines, parsed {current_hunk.old_seen} old/"
                f"{current_hunk.new_seen} new lines.",
            )
        if current_file is not None:
            current_file.hunks.append(current_hunk.build())
        current_hunk = None

    for input_line_number, line in enumerate(input_lines, start=1):
        if current_hunk is not None and current_hunk.complete:
            finish_hunk()

        if current_hunk is not None:
            if line == "\\ No newline at end of file":
                continue
            if current_hunk.add_line(line):
                continue
            finish_hunk()

        if pending_old_header and not line.startswith("+++ "):
            _add_warning(
                warnings,
                "Unified diff file header at input line "
                f"{input_line_number - 1} is missing a following +++ header.",
            )
            pending_old_header = False

        if line.startswith("--- "):
            pending_old_path, pending_old_path_valid = _normalize_header_path(
                line,
                warnings,
                input_line_number,
            )
            pending_old_header = True
            current_file = None
            continue

        if line.startswith("+++ "):
            if not pending_old_header:
                _add_warning(
                    warnings,
                    f"Unexpected +++ header at input line {input_line_number}.",
                )
                continue

            new_path, new_path_valid = _normalize_header_path(
                line,
                warnings,
                input_line_number,
            )
            pending_old_header = False
            if (
                not pending_old_path_valid
                or not new_path_valid
                or (pending_old_path is None and new_path is None)
            ):
                current_file = None
                continue

            if len(changed_files) >= MAX_DIFF_FILES:
                _add_warning(
                    warnings,
                    f"Changed files truncated at MAX_DIFF_FILES={MAX_DIFF_FILES}.",
                )
                current_file = None
                continue

            current_file = ChangedFile(
                old_path=pending_old_path,
                new_path=new_path,
            )
            changed_files.append(current_file)
            if (
                pending_old_path is not None
                and new_path is not None
                and pending_old_path != new_path
            ):
                _add_warning(
                    warnings,
                    "Rename has limited support: "
                    f"{pending_old_path} -> {new_path}.",
                )
            continue

        if line.startswith("@@"):
            if current_file is None:
                _add_warning(
                    warnings,
                    f"Hunk at input line {input_line_number} has no file header.",
                )
                continue
            current_hunk = _parse_hunk_header(line, input_line_number)
            if current_hunk is None:
                _add_warning(
                    warnings,
                    f"Malformed hunk header at input line {input_line_number}.",
                )
            continue

        if line.startswith(("Binary files ", "GIT binary patch")):
            if current_file is not None:
                current_file.is_binary = True
            _add_warning(warnings, "Binary diff content was not parsed.")

    finish_hunk()
    if pending_old_header:
        _add_warning(
            warnings,
            "Unified diff ended before a matching +++ file header.",
        )

    matched_index: int | None = None
    normalized_target: str | None = None
    if target_file is not None:
        matched_index = _target_match(changed_files, target_file)
        if matched_index is None:
            normalized_target = _fallback_target_path(target_file)
            _add_warning(
                warnings,
                "Target file was not found in supplied diff.",
            )
        else:
            matched_file = changed_files[matched_index]
            normalized_target = matched_file.new_path or matched_file.old_path

    target_hunks: list[DiffHunk] = []
    if matched_index is not None:
        all_target_hunks = changed_files[matched_index].hunks
        target_hunks = all_target_hunks[:MAX_TARGET_HUNKS]
        if len(all_target_hunks) > MAX_TARGET_HUNKS:
            _add_warning(
                warnings,
                "Target hunks truncated at "
                f"MAX_TARGET_HUNKS={MAX_TARGET_HUNKS}.",
            )

    changed_new_lines = sorted(
        {
            line.new_line_number
            for hunk in target_hunks
            for line in hunk.lines
            if line.kind == "add" and line.new_line_number is not None
        }
    )
    return DiffContext(
        target_file=normalized_target,
        changed_files=changed_files,
        target_hunks=target_hunks,
        changed_new_lines=changed_new_lines,
        changed_new_line_ranges=_line_ranges(changed_new_lines),
        warnings=warnings,
    )
