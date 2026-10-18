"""Anthropic API call: builds the user message, sends it, and parses the JSON reply."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from importlib import resources

import anthropic
from anthropic.types import TextBlock

from codeverify.config import Config
from codeverify.errors import ConfigError, LLMReviewError
from codeverify.models import ASTFinding

logger = logging.getLogger(__name__)

TEMPERATURE = 0.1
MAX_TOKENS = 4096
MAX_DIFF_CHARS = 500_000
_REQUEST_TIMEOUT_SECONDS = 120.0
_MAX_RETRIES = 3
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def load_system_prompt() -> str:
    """Read prompts/system.txt from the installed package."""
    try:
        return resources.files("codeverify").joinpath("prompts/system.txt").read_text(
            encoding="utf-8"
        )
    except (OSError, ModuleNotFoundError) as exc:
        raise LLMReviewError(f"Cannot load system prompt: {exc}") from exc


def build_user_message(diff: str, ast_findings: Sequence[ASTFinding], description: str) -> str:
    ast_json = json.dumps([finding.model_dump() for finding in ast_findings], indent=2)
    return f"[DIFF]\n{diff}\n\n[AST_FINDINGS]\n{ast_json}\n\n[PR_DESCRIPTION]\n{description}"


def parse_json_response(text: str) -> object:
    """Parse the model reply as JSON, tolerating a stray code fence or surrounding prose."""
    candidate = text.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMReviewError("Model response was not valid JSON.")


def request_review(
    diff: str,
    ast_findings: Sequence[ASTFinding],
    description: str,
    config: Config,
    client: anthropic.Anthropic | None = None,
) -> object:
    """Send the review request and return the parsed (still unvalidated) JSON value."""
    if len(diff) > MAX_DIFF_CHARS:
        raise LLMReviewError(
            f"Diff is {len(diff):,} characters (limit {MAX_DIFF_CHARS:,}). "
            "Add ignore_patterns for generated or vendored files, or review a smaller range."
        )

    if client is None:
        try:
            api_key = config.anthropic_api_key()
        except ConfigError as exc:
            raise LLMReviewError(str(exc)) from exc
        client = anthropic.Anthropic(
            api_key=api_key, max_retries=_MAX_RETRIES, timeout=_REQUEST_TIMEOUT_SECONDS
        )

    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=[
                {
                    "type": "text",
                    "text": load_system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": build_user_message(diff, ast_findings, description)}
            ],
        )
    except anthropic.APIError as exc:
        raise LLMReviewError(f"Anthropic API request failed: {exc}") from exc

    logger.debug(
        "Usage: input=%s output=%s cache_read=%s cache_write=%s",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.cache_read_input_tokens,
        response.usage.cache_creation_input_tokens,
    )
    if response.stop_reason == "max_tokens":
        raise LLMReviewError(
            f"Model output was cut off at {MAX_TOKENS} tokens, so the JSON is incomplete. "
            "Review a smaller diff."
        )

    text = "".join(block.text for block in response.content if isinstance(block, TextBlock))
    if not text.strip():
        raise LLMReviewError("Model returned no text content.")
    return parse_json_response(text)
