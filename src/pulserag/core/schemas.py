"""Data shapes shared across the application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ConfidenceLevel = Literal["high", "moderate", "low"]


class Citation(BaseModel):
    """Retrieved chunk with source and relevance score."""

    label: str
    source_org: str
    score: float | None


class RAGResponse(BaseModel):
    """Answer with supporting evidence."""

    answer: str
    evidence: str
    citations: list[Citation]
    confidence: ConfidenceLevel
    disclaimer: str = Field(description="Project disclaimer")


class ReadinessCheck(BaseModel):
    """Result of one dependency probe."""

    name: str
    healthy: bool
    details: str | None = Field(description="Failure Reason or Context when healthy.")
