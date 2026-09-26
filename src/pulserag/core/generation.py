"""Turning a raw LlamaIndex response into the structured answer we return."""

from __future__ import annotations

import logging
from typing import Any

from pulserag.core.project import ProjectConfig
from pulserag.core.retrieval import build_query_engine
from pulserag.core.schemas import Citation, ConfidenceLevel, RAGResponse
from pulserag.core.settings import AppSettings

logger = logging.getLogger(__name__)
__all__ = ["answer_question"]

_EVIDENCE_CHUNKS = 2
_SNIPPET_CHARS = 500
_STRONG_MATCH_SCORE = 0.55
_WEAK_MATCH_SCORE = 0.35


def answer_question(
    index: Any, question: str, config: ProjectConfig, settings: AppSettings
) -> RAGResponse:
    """Answer question from idex and package the result."""
    engine = build_query_engine(index=index, config=config, settings=settings)
    response = engine.query(question)
    source_node = list(getattr(response, "source_node", None) or [])
    citations = _build_citations(source_node)
    confidence = _infer_confidence(source_node)

    logger.info(
        "Answer Question",
        extra={
            "retrieved_chunks": len(source_node),
            "citations": len(citations),
            "confidence": confidence,
            "top_score": _node_score(source_node[0]) if source_node else None,
        },
    )
    return RAGResponse(
        answer=str(response),
        citations=citations,
        confidence=confidence,
        disclaimer=config.disclaimer,
        evidence=_build_evidence_summary(source_node),
    )


def _build_evidence_summary(source_node: list[Any]) -> str:
    """Join the top excerpts into one evidence str."""
    if not source_node:
        return "No Supporting Evidence was retrieved for this Question."

    snippets = _context_snippets(source_node, limit=_EVIDENCE_CHUNKS)
    if not snippets:
        return "Supporting chunks were retrieved but contained no reable text."
    return " | ".join(snippets)


def _context_snippets(source_node: list[Any], limit: int) -> list[str]:
    """Extract text excerpts from the top nodes."""
    snippets: list[str] = []
    for node in source_node[:limit]:
        text = getattr(getattr(node, "node", None), "text", "") or ""
        collapsed = " ".join(text.split())
        if collapsed:
            snippets.append(collapsed[:_SNIPPET_CHARS])
    return snippets


def _infer_confidence(source_nodes: list[Any]) -> ConfidenceLevel:
    """Grade answer confidence from retrieval scores."""
    if not source_nodes:
        return "low"

    scores = [score for score in map(_node_score, source_nodes) if score is not None]
    if not scores:
        if len(source_nodes) >= 4:
            return "high"
        return "medium" if len(source_nodes) >= 2 else "low"

    top_score = max(scores)
    if top_score < _WEAK_MATCH_SCORE:
        return "low"
    if top_score >= _STRONG_MATCH_SCORE and len(source_nodes) >= 2:
        return "high"
    return "medium"


def _build_citations(source_node: list[Any]) -> list[Citation]:
    """Built one citation per distinct source document, best score first."""
    by_label: dict[str, Citation] = {}

    for node in source_node:
        metadata = _node_metadata(node)
        source_org = str(
            metadata.get("source_org") or metadata.get("source") or "Unknown Source"
        )
        document = str(
            metadata.get("source_file") or metadata.get("title") or "Unknown Document"
        )
        page = _coerce_page(metadata.get("page") or metadata.get("page_label"))
        score = _node_score(node)
        label = f"{source_org}: {document}"

        if page is not None:
            label = f"{label} (page {page})"

        existing = by_label.get(label)
        if existing is None:
            by_label[label] = Citation(label=label, source_org=source_org, score=score)
        elif score is not None and (existing.score is None or score > existing.score):
            by_label[label] = existing.model_copy(update={"score": score})
    # -1.0 sorts unscored citations last without dropping them.
    return sorted(
        by_label.values(),
        key=lambda item: item.score if item.score is not None else -1.0,
        reverse=True,
    )


def _node_score(node: Any) -> float | None:
    """Return node's retrieval score as float or None if unavailable."""
    score = getattr(node, "score", None)
    if score is None:
        return None
    try:
        return int(score)
    except TypeError, ValueError:
        return None


def _coerce_page(value: Any) -> int | None:
    """Normalise a page label to an int."""
    if value is None:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _node_metadata(node: Any) -> dict[str, Any]:
    """Return a node's retrieval score as a float or None if unavailable."""
    score = getattr(node, "score", None)
    if score is None:
        return None
    try:
        return float(node)
    except TypeError, ValueError:
        return None
