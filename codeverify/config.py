"""Configuration: config.yaml first, environment variables as fallback, then defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from codeverify.errors import ConfigError

DEFAULT_CONFIG_FILE: Final = "config.yaml"
DEFAULT_MODEL: Final = "claude-sonnet-4-6"
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("python", "javascript", "typescript", "go", "rust")

_ENV_VARS: Final[dict[str, str]] = {
    "model": "CODEVERIFY_MODEL",
    "confidence_threshold": "CODEVERIFY_CONFIDENCE_THRESHOLD",
    "languages": "CODEVERIFY_LANGUAGES",
    "ignore_patterns": "CODEVERIFY_IGNORE_PATTERNS",
    "anthropic_api_key_env": "CODEVERIFY_ANTHROPIC_API_KEY_ENV",
}


@dataclass(frozen=True)
class Config:
    model: str = DEFAULT_MODEL
    confidence_threshold: float = 0.80
    languages: tuple[str, ...] = SUPPORTED_LANGUAGES
    ignore_patterns: tuple[str, ...] = ()
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"

    def anthropic_api_key(self) -> str:
        """Return the API key from the environment variable named by the config."""
        key = os.environ.get(self.anthropic_api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"Environment variable {self.anthropic_api_key_env} is not set; "
                "it must contain your Anthropic API key."
            )
        return key


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    """Load configuration.

    Precedence per field: config file value, then environment variable, then default.
    An explicit `path` must exist; the implicit ./config.yaml is optional.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    file_values = _read_config_file(path)

    unknown = sorted(set(file_values) - set(_ENV_VARS))
    if unknown:
        raise ConfigError(
            f"Unknown config key(s): {', '.join(unknown)}. Valid keys: {', '.join(_ENV_VARS)}."
        )

    raw: dict[str, object] = {}
    for name, env_var in _ENV_VARS.items():
        if file_values.get(name) is not None:
            raw[name] = file_values[name]
        elif environ.get(env_var, "").strip():
            raw[name] = environ[env_var].strip()
    return _build_config(raw)


def _read_config_file(path: Path | None) -> dict[str, object]:
    target = path if path is not None else Path(DEFAULT_CONFIG_FILE)
    if not target.is_file():
        if path is not None:
            raise ConfigError(f"Config file not found: {path}")
        return {}
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read config file {target}: {exc}") from exc
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {target}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{target} must contain a YAML mapping at the top level.")
    return {str(key): value for key, value in data.items()}


def _build_config(raw: Mapping[str, object]) -> Config:
    defaults = Config()
    return Config(
        model=_as_string(raw.get("model", defaults.model), "model"),
        confidence_threshold=_as_threshold(
            raw.get("confidence_threshold", defaults.confidence_threshold)
        ),
        languages=_as_languages(raw.get("languages", defaults.languages)),
        ignore_patterns=_as_string_list(
            raw.get("ignore_patterns", defaults.ignore_patterns), "ignore_patterns"
        ),
        anthropic_api_key_env=_as_string(
            raw.get("anthropic_api_key_env", defaults.anthropic_api_key_env),
            "anthropic_api_key_env",
        ),
    )


def _as_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _as_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError("'confidence_threshold' must be a number between 0 and 1.")
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigError("'confidence_threshold' must be a number between 0 and 1.") from exc
    if not 0.0 <= number <= 1.0:
        raise ConfigError("'confidence_threshold' must be between 0 and 1.")
    return number


def _as_string_list(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        items: list[object] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ConfigError(f"'{name}' must be a list of strings (or a comma-separated string).")
    result: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"'{name}' entries must be non-empty strings.")
        result.append(item.strip())
    return tuple(result)


def _as_languages(value: object) -> tuple[str, ...]:
    languages = tuple(lang.lower() for lang in _as_string_list(value, "languages"))
    unsupported = sorted(set(languages) - set(SUPPORTED_LANGUAGES))
    if unsupported:
        raise ConfigError(
            f"Unsupported language(s): {', '.join(unsupported)}. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}."
        )
    return languages
