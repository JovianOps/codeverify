"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["BLOCKER", "WARNING", "ADVISORY"]
Category = Literal[
    "hallucinated_api",
    "injection",
    "secret",
    "missing_error_handling",
    "race_condition",
    "off_by_one",
    "null_deref",
    "resource_leak",
    "logic_error",
    "performance",
]


class ASTFinding(BaseModel):
    """A hint produced by the tree-sitter pass (not a verdict)."""

    model_config = ConfigDict(frozen=True)

    file: str
    line: int
    symbol: str
    kind: str


class Finding(BaseModel):
    """A single defect reported by the model."""

    model_config = ConfigDict(extra="ignore")

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    severity: Severity
    category: Category
    title: str = Field(min_length=1)
    explanation: str
    suggested_fix: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewResult(BaseModel):
    findings: list[Finding]
    summary: str
