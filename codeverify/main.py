"""Pipeline orchestration: diff -> AST hints -> LLM review -> validated result."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from codeverify.ast_analyzer import ProjectContext, analyze_files
from codeverify.config import Config
from codeverify.diff_parser import is_ignored, parse_diff, render_diff, require_diff_like
from codeverify.errors import DiffSourceError, ResponseValidationError
from codeverify.llm_reviewer import request_review
from codeverify.models import ReviewResult
from codeverify.validator import validate_response

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECONDS = 60


def run_review(
    diff_text: str, description: str, config: Config, context: ProjectContext
) -> ReviewResult:
    """Run the full pipeline on a unified diff."""
    require_diff_like(diff_text)
    files = parse_diff(diff_text)
    reviewable = [f for f in files if not is_ignored(f.path, config.ignore_patterns)]
    skipped = len(files) - len(reviewable)
    if skipped:
        logger.info("Ignored %d file(s) matching ignore_patterns", skipped)
    if not reviewable:
        return ReviewResult(
            findings=[], summary="No reviewable changes: the diff has no added or modified text files."
        )

    ast_findings = analyze_files(reviewable, config.languages, context)
    logger.info("AST pass produced %d hint(s)", len(ast_findings))

    raw = request_review(render_diff(reviewable), ast_findings, description, config)
    return validate_response(raw, reviewable, config.confidence_threshold)


def load_diff(source: str, repo_root: Path) -> str:
    """Read a diff from a file, from stdin ('-'), or from git.

    A git ref `A..B` / `A...B` is diffed as a range; a single ref is diffed against its parent.
    """
    if source == "-":
        return sys.stdin.read()
    path = Path(source)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DiffSourceError(f"Cannot read diff file {path}: {exc}") from exc
    return _git_diff(source, repo_root)


def load_result(path: Path) -> ReviewResult:
    """Load a ReviewResult JSON file produced by `codeverify diff`."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResponseValidationError(f"Cannot read result file {path}: {exc}") from exc
    try:
        return ReviewResult.model_validate_json(content)
    except ValidationError as exc:
        raise ResponseValidationError(f"{path} is not a valid review result: {exc}") from exc


def _git_diff(ref: str, repo_root: Path) -> str:
    if not ref.strip() or ref.startswith("-"):
        raise DiffSourceError(f"'{ref}' is neither a diff file nor a valid git ref.")
    base = ["git", "diff", "--no-color", "--no-ext-diff"]
    command = [*base, ref, "--"] if ".." in ref else [*base, f"{ref}^", ref, "--"]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DiffSourceError("git is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise DiffSourceError(f"git diff timed out after {_GIT_TIMEOUT_SECONDS}s.") from exc
    if completed.returncode != 0:
        raise DiffSourceError(
            f"'{ref}' is not a diff file and `git diff` failed: {completed.stderr.strip()}"
        )
    return completed.stdout
