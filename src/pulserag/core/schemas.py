"""Data shapes shared across the application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ConfidenceLevel = Literal["high", "medium", "low"]


class Citation(BaseModel):
    """Retrieved chunk with source and relevance score."""

    label: str
    source_org: str
    score: float | None = Field(default=None)


class RAGResponse(BaseModel):
    """Answer with supporting evidence."""

    answer: str
    evidence: str
    citations: list[Citation]
    confidence: ConfidenceLevel
    disclaimer: str = Field(description="Project disclaimer")


@dataclass(frozen=True)
class QueryArtifacts:
    """Everything one query produced, for caller that need more details."""

    response: RAGResponse
    retrieval_context: list[str]


class ReadinessCheck(BaseModel):
    """Result of one dependency probe."""

    name: str
    healthy: bool
    details: str | None = Field(description="Failure Reason or Context when healthy.")


class EvalMetricResult(BaseModel):
    """Score for one metric on one test case."""

    name: str = Field(description="Display Name, e.g. 'Groundedness'")
    score: float | None
    threshold: float | None
    success: bool
    reason: str | None = Field(default=None)
    error: str | None = Field(default=None)


class EvalCaseResult(BaseModel):
    """One Golden Dataset case, its answer and every metric applied to it."""

    id: str
    category: str | None = Field(default=None)
    query: str
    expected_answer: str
    actual_answer: str
    sources: list[str]
    retrieval_context: list[str]
    metrics: list[EvalMetricResult]
    success: bool


class EvalSummary(BaseModel):
    """Result of one evaluation run."""

    project: str
    collection_name: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    dataset_size: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    skipped_case_ids: list[str]
    success_rate: float
    success: bool


class EvalReport(BaseModel):
    """Full Eval Report."""

    summary: EvalSummary
    cases: list[EvalCaseResult]
