"""Unified-diff parsing: per-file new-side line numbers, added lines, and ignore filtering."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

from codeverify.errors import DiffSourceError

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_ESCAPES = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11, "\\": 92, '"': 34}
_OCTAL_DIGITS = "01234567"


@dataclass
class FileDiff:
    """One file's changes on the new side of a diff."""

    path: str
    raw: str
    added: dict[int, str]
    context: dict[int, str]

    def new_side(self) -> dict[int, str]:
        """Every new-side line present in the diff (added + context), by line number."""
        return {**self.context, **self.added}

    def is_added_line(self, line: int) -> bool:
        return line in self.added


@dataclass
class _Section:
    raw: list[str] = field(default_factory=list)
    path: str | None = None
    saw_old: bool = False
    saw_new: bool = False
    added: dict[int, str] = field(default_factory=dict)
    context: dict[int, str] = field(default_factory=dict)

    def to_file_diff(self) -> FileDiff | None:
        if self.path is None or not (self.added or self.context):
            return None
        return FileDiff(self.path, "\n".join(self.raw) + "\n", self.added, self.context)


def parse_diff(text: str) -> list[FileDiff]:
    """Parse a unified diff (git or plain `diff -u`). Deleted files and binary files are skipped."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    files: list[FileDiff] = []
    section = _Section()
    old_remaining = 0
    new_remaining = 0
    new_line = 0
    index = 0

    def flush() -> None:
        parsed = section.to_file_diff()
        if parsed is not None:
            files.append(parsed)

    while index < len(lines):
        line = lines[index]

        if old_remaining > 0 or new_remaining > 0:
            if line.startswith("+"):
                section.added[new_line] = line[1:]
                new_line += 1
                new_remaining -= 1
            elif line.startswith("-"):
                old_remaining -= 1
            elif line.startswith(" ") or line == "":
                section.context[new_line] = line[1:]
                new_line += 1
                new_remaining -= 1
                old_remaining -= 1
            elif not line.startswith("\\"):
                # Malformed or truncated hunk: end it and re-read this line as a header.
                old_remaining = 0
                new_remaining = 0
                continue
            section.raw.append(line)
            index += 1
            continue

        hunk = _HUNK_RE.match(line)
        if line.startswith("diff --git "):
            flush()
            section = _Section(raw=[line])
        elif line.startswith("--- "):
            if section.saw_new:
                flush()
                section = _Section()
            section.saw_old = True
            section.raw.append(line)
        elif line.startswith("+++ ") and section.saw_old and not section.saw_new:
            section.saw_new = True
            section.path = _parse_path(line[4:])
            section.raw.append(line)
        elif hunk is not None and section.saw_new:
            old_remaining = int(hunk.group(2)) if hunk.group(2) is not None else 1
            new_line = int(hunk.group(3))
            new_remaining = int(hunk.group(4)) if hunk.group(4) is not None else 1
            section.raw.append(line)
        elif section.raw:
            section.raw.append(line)
        index += 1

    flush()
    return files


def render_diff(files: Sequence[FileDiff]) -> str:
    """Re-assemble the raw diff text for the given files."""
    return "".join(file_diff.raw for file_diff in files)


def is_ignored(path: str, patterns: Sequence[str]) -> bool:
    """Glob match (case-sensitive). `**/x` also matches a top-level `x`; `dir/` ignores a tree."""
    for pattern in patterns:
        candidates = {pattern}
        if pattern.startswith("**/"):
            candidates.add(pattern[3:])
        if pattern.endswith("/"):
            candidates.add(f"{pattern}*")
        if any(fnmatchcase(path, candidate) for candidate in candidates):
            return True
    return False


def require_diff_like(text: str) -> None:
    """Reject input that clearly is not a unified diff (e.g. a wrong path or an HTML error page)."""
    if text.strip() and "@@" not in text and "diff --git" not in text:
        raise DiffSourceError("Input does not look like a unified diff (no hunks found).")


def _parse_path(token: str) -> str | None:
    """Extract the new-side path from a ---/+++ header value; None for /dev/null."""
    value = token.rstrip("\r").split("\t")[0]
    if value == "/dev/null":
        return None
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = _unquote_git_path(value[1:-1])
    for prefix in ("b/", "a/"):
        if value.startswith(prefix):
            return value[len(prefix):] or None
    return value or None


def _unquote_git_path(value: str) -> str:
    out = bytearray()
    i = 0
    while i < len(value):
        char = value[i]
        if char == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt in _OCTAL_DIGITS:
                j = i + 1
                while j < len(value) and j < i + 4 and value[j] in _OCTAL_DIGITS:
                    j += 1
                out.append(int(value[i + 1 : j], 8) & 0xFF)
                i = j
                continue
            if nxt in _GIT_ESCAPES:
                out.append(_GIT_ESCAPES[nxt])
            else:
                out.extend(nxt.encode("utf-8"))
            i += 2
            continue
        out.extend(char.encode("utf-8"))
        i += 1
    return out.decode("utf-8", errors="replace")
