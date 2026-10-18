"""Validate the model's JSON against the schema, the confidence threshold, and the diff."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from codeverify.diff_parser import FileDiff
from codeverify.errors import ResponseValidationError
from codeverify.models import Finding, ReviewResult

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"BLOCKER": 0, "WARNING": 1, "ADVISORY": 2}


def validate_response(
    data: object, files: Sequence[FileDiff], threshold: float
) -> ReviewResult:
    """Return a ReviewResult containing only findings that are well-formed, confident enough,
    and anchored to an added line in the diff.

    Raises ResponseValidationError only when the top-level structure is unusable; a malformed
    individual finding is dropped with a warning instead of discarding the whole review.
    """
    if not isinstance(data, dict):
        raise ResponseValidationError("Model JSON must be an object with 'findings' and 'summary'.")
    raw_findings = data.get("findings")
    summary = data.get("summary")
    if not isinstance(raw_findings, list):
        raise ResponseValidationError("Model JSON field 'findings' must be a list.")
    if not isinstance(summary, str):
        raise ResponseValidationError("Model JSON field 'summary' must be a string.")

    by_path = {file_diff.path: file_diff for file_diff in files}
    accepted: dict[tuple[str, int, str, str], Finding] = {}

    for position, item in enumerate(raw_findings):
        try:
            finding = Finding.model_validate(item)
        except ValidationError as exc:
            logger.warning("Dropping finding #%d: schema violation: %s", position, exc.errors())
            continue

        if finding.confidence < threshold:
            logger.debug("Dropping finding #%d: confidence %.2f < %.2f", position, finding.confidence, threshold)
            continue

        path = _resolve_path(finding.file, by_path)
        if path is None:
            logger.warning("Dropping finding #%d: file %r is not in the diff", position, finding.file)
            continue
        if not by_path[path].is_added_line(finding.line):
            logger.warning(
                "Dropping finding #%d: %s:%d is not an added line in the diff",
                position,
                path,
                finding.line,
            )
            continue

        canonical = finding.model_copy(update={"file": path})
        key = (path, canonical.line, canonical.category, canonical.title)
        accepted.setdefault(key, canonical)

    findings = sorted(
        accepted.values(),
        key=lambda f: (_SEVERITY_ORDER[f.severity], f.file, f.line),
    )
    dropped = len(raw_findings) - len(findings)
    if dropped:
        logger.info("Kept %d of %d findings", len(findings), len(raw_findings))
    return ReviewResult(findings=findings, summary=summary)


def _resolve_path(path: str, by_path: Mapping[str, FileDiff]) -> str | None:
    """Match the model's path to a diff path, tolerating `./` and `a/`/`b/` prefixes."""
    candidates = [path, path.removeprefix("./"), path.removeprefix("b/"), path.removeprefix("a/")]
    for candidate in candidates:
        if candidate in by_path:
            return candidate
    return None
