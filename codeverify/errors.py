"""Exception hierarchy. The CLI turns any CodeVerifyError into a clean error message."""

from __future__ import annotations


class CodeVerifyError(Exception):
    """Base class for all expected, user-facing failures."""


class ConfigError(CodeVerifyError):
    """Configuration file or environment is invalid."""


class DiffSourceError(CodeVerifyError):
    """The diff could not be read from a file, stdin, or git."""


class LLMReviewError(CodeVerifyError):
    """The Anthropic API call failed or returned unusable output."""


class ResponseValidationError(CodeVerifyError):
    """The model response does not have the required top-level structure."""


class GitHubError(CodeVerifyError):
    """A GitHub API call failed."""
