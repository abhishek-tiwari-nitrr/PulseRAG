"""The quality metrics every evaluation run applies.

======================  ====================================================
Metric                  Question it answers
======================  ====================================================
Answer Relevance        Did we answer the question that was asked?
Context Relevance       Did the retriever fetch material that was on-topic?
Groundedness            Is every claim in the answer supported by the context? (also catches hallucination)
======================  ====================================================

"""

from __future__ import annotations

import contextlib
from typing import Any

__all__ = []
DEFAULT_METRIC_THRESHOLD = 0.7
_DISPLAY_NAMES = {
    "AnswerRelevancyMetric": "Answer Relevance",
    "ContextualRelevancyMetric": "Context Relevance",
    "FaithfulnessMetric": "Groundedness",
}


def build_rag_metrics(threshold: float = DEFAULT_METRIC_THRESHOLD) -> list[Any]:
    """Construct a fresh set of metric objects."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    return [
        _with_display_name(AnswerRelevancyMetric(threshold=threshold)),
        _with_display_name(ContextualRelevancyMetric(threshold=threshold)),
        _with_display_name(FaithfulnessMetric(threshold=threshold)),
    ]


def _with_display_name(metric: Any) -> Any:
    """Tag a metric with name and return it."""
    label = _DISPLAY_NAMES.get(type(metric).__name__)
    if label is not None:
        with contextlib.suppress(AttributeError):
            metric.name = label
    return metric
