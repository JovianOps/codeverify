"""GitHub boundary: open a repo, fetch a PR as a unified diff, and post the review."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from github import Auth, Github
from github.GithubException import GithubException
from github.Repository import Repository

from codeverify.errors import GitHubError
from codeverify.models import Finding, ReviewResult

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.github.com"
MAX_BODY_CHARS = 60_000
_MAX_FILE_BYTES = 1_000_000
_REQUEST_TIMEOUT_SECONDS = 30
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9_-])")
_BACKTICK_RUN_RE = re.compile(r"`+")


@dataclass(frozen=True)
class PullRequestData:
    diff: str
    description: str
    head_sha: str


@dataclass(frozen=True)
class ReviewPayload:
    event: str
    body: str
    comments: list[dict[str, Any]]


def open_repository(full_name: str) -> Repository:
    """Authenticate with GITHUB_TOKEN (and GITHUB_API_URL for Enterprise) and open the repo."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise GitHubError("Environment variable GITHUB_TOKEN is not set.")
    base_url = os.environ.get("GITHUB_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
    client = Github(auth=Auth.Token(token), base_url=base_url, timeout=_REQUEST_TIMEOUT_SECONDS)
    try:
        return client.get_repo(full_name)
    except GithubException as exc:
        raise GitHubError(f"Cannot open repository {full_name}: {_describe(exc)}") from exc


def fetch_pull_request(repo: Repository, number: int) -> PullRequestData:
    """Rebuild a git-style unified diff from the PR's per-file patches."""
    try:
        pull = repo.get_pull(number)
        chunks: list[str] = []
        for changed in pull.get_files():
            if changed.status == "removed":
                continue
            if not changed.patch:
                logger.warning("No textual patch for %s (binary or too large); skipping", changed.filename)
                continue
            old_path = changed.previous_filename or changed.filename
            old_header = "/dev/null" if changed.status == "added" else f"a/{old_path}"
            chunks.append(
                f"diff --git a/{old_path} b/{changed.filename}\n"
                f"--- {old_header}\n"
                f"+++ b/{changed.filename}\n"
                f"{changed.patch}\n"
            )
        description = f"{pull.title}\n\n{pull.body or ''}".strip()
        return PullRequestData(diff="".join(chunks), description=description, head_sha=pull.head.sha)
    except GithubException as exc:
        raise GitHubError(f"Cannot fetch PR #{number}: {_describe(exc)}") from exc


class GitHubProjectContext:
    """ProjectContext that reads files from the PR head commit through the Contents API."""

    def __init__(self, repo: Repository, ref: str) -> None:
        self._repo = repo
        self._ref = ref
        self._exists: dict[str, bool] = {}

    def read_file(self, path: str) -> str | None:
        try:
            item = self._repo.get_contents(path, ref=self._ref)
        except GithubException as exc:
            if exc.status != 404:
                logger.warning("Cannot read %s at %s: %s", path, self._ref, _describe(exc))
            return None
        if isinstance(item, list) or item.encoding != "base64" or item.size > _MAX_FILE_BYTES:
            return None
        try:
            return item.decoded_content.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def path_exists(self, path: str) -> bool:
        if path not in self._exists:
            try:
                self._repo.get_contents(path, ref=self._ref)
                self._exists[path] = True
            except GithubException as exc:
                if exc.status != 404:
                    logger.warning("Cannot stat %s at %s: %s", path, self._ref, _describe(exc))
                self._exists[path] = False
        return self._exists[path]


def build_review_payload(result: ReviewResult) -> ReviewPayload:
    """REQUEST_CHANGES if any BLOCKER, else COMMENT; body is the summary; one inline comment each."""
    has_blocker = any(finding.severity == "BLOCKER" for finding in result.findings)
    body = _sanitize(result.summary).strip() or "codeverify: no summary provided."
    comments = [
        {
            "path": finding.file,
            "line": finding.line,
            "side": "RIGHT",
            "body": _comment_body(finding),
        }
        for finding in result.findings
    ]
    return ReviewPayload(
        event="REQUEST_CHANGES" if has_blocker else "COMMENT",
        body=body[:MAX_BODY_CHARS],
        comments=comments,
    )


def post_review(
    repo: Repository, number: int, result: ReviewResult, commit_sha: str | None = None
) -> str:
    """Post the review and return its URL.

    If GitHub rejects the inline comments (422: a line is not part of the PR diff, or you are
    requesting changes on your own PR) the findings are posted in the review body instead.
    """
    payload = build_review_payload(result)
    try:
        pull = repo.get_pull(number)
        commit = repo.get_commit(commit_sha or pull.head.sha)
        try:
            review = pull.create_review(
                commit=commit, body=payload.body, event=payload.event, comments=payload.comments
            )
        except GithubException as exc:
            if exc.status != 422:
                raise
            logger.warning("GitHub rejected the inline review (%s); posting findings in the body", _describe(exc))
            review = pull.create_review(
                commit=commit, body=_fallback_body(result), event="COMMENT"
            )
        return str(review.html_url)
    except GithubException as exc:
        raise GitHubError(f"Cannot post review to PR #{number}: {_describe(exc)}") from exc


def _comment_body(finding: Finding) -> str:
    lines = [f"**[{finding.severity}] {_sanitize(finding.title)}**", "", _sanitize(finding.explanation)]
    fix = finding.suggested_fix.strip("\n")
    if fix.strip():
        fence = _fence_for(fix)
        lines += ["", "**Suggested fix:**", fence, fix, fence]
    lines += ["", f"<sub>{finding.category} · confidence {finding.confidence:.2f}</sub>"]
    return "\n".join(lines)[:MAX_BODY_CHARS]


def _fallback_body(result: ReviewResult) -> str:
    lines = [_sanitize(result.summary).strip() or "codeverify review", ""]
    if result.findings:
        lines += ["---", "**Findings** (could not be attached inline):", ""]
        for finding in result.findings:
            lines.append(
                f"- `{finding.file}:{finding.line}` **[{finding.severity}]** "
                f"{_sanitize(finding.title)}: {_sanitize(finding.explanation)}"
            )
    return "\n".join(lines)[:MAX_BODY_CHARS]


def _fence_for(text: str) -> str:
    longest = max((len(run) for run in _BACKTICK_RUN_RE.findall(text)), default=0)
    return "`" * max(3, longest + 1)


def _sanitize(text: str) -> str:
    """Defuse @mentions: model output is influenced by untrusted diff content."""
    return _MENTION_RE.sub("@\u200b", text)


def _describe(exc: GithubException) -> str:
    return f"HTTP {exc.status}: {exc.data}"
