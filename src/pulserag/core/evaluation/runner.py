"""Running the golden-dataset evaluation and persisting its results.

Steps:
1. Read golden dataset.
2. Ask the live pipeline each question, exactly as a user would.
3. Scores each answer with the metric.
4. Write full report to data dir, so the last run can be read back without re-running it.

"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pulserag.core.evaluation.metrics import build_rag_metrics
from pulserag.core.exceptions import IndexNotReadyError
from pulserag.core.project import ProjectConfig
from pulserag.core.schemas import (
    EvalCaseResult,
    EvalMetricResult,
    EvalReport,
    EvalSummary,
)
from pulserag.core.settings import AppSettings

if TYPE_CHECKING:
    from pulserag.core.service import RAGService

logger = logging.getLogger(__name__)

__all__ = ["load_latest_eval_result", "run_evaluation"]
SOURCE_AVAILABILITY: dict[str, Callable[[AppSettings], bool]] = {
    "guideline_pdf": lambda s: s.max_guideline_files > 0,
    "pubmed": lambda s: s.pubmed_enabled and s.pubmed_query_limit > 0,
    "bootstrap": lambda s: s.include_bootstrap_documents,
}
REQUIRED_CASE_KEYS = frozenset({"id", "query", "expected_answer"})


def load_eval_dataset(config: ProjectConfig):
    """Load golden dataset from the folder."""
    path = config.golden_dataset_path
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}.")

    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError(f"Golden dataset at {path} must be json list of case objects.")

    missing = [
        case.get("id", f"#{position}") if isinstance(case, dict) else f"#{position}"
        for position, case in enumerate(dataset)
        if not isinstance(case, dict) or not set(case) >= REQUIRED_CASE_KEYS
    ]

    if missing:
        raise ValueError(
            f"Golden dataset cases are missing required keys (id, query, expected_answer): {', '.join(map(str, missing))}"
        )
    return dataset


def case_is_applicable(case: dict[str, Any], settings: AppSettings) -> bool:
    """Whether every source a case `requires` was part of this index."""
    return all(
        SOURCE_AVAILABILITY[name](settings) for name in case.get("requires") or []
    )


def latest_result_path(config: ProjectConfig) -> Path:
    """Path of the JSON file holding most recent run."""
    return config.eval_result_dir / f"{config.name}_latest.json"


def load_latest_eval_result(config: ProjectConfig) -> EvalReport | None:
    """Load the most recent persisted evaluation report."""
    path = latest_result_path(config)
    if not path.exists():
        return None
    try:
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning(
            "Stored evaluation report could not be parsed; may it is absent.",
            extra={"path": str(path)},
            exc_info=True,
        )
        return None


def save_latest_eval_result(config: ProjectConfig, result: EvalReport) -> Path:
    """Report of the latest result."""
    path = latest_result_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.temp")
    temp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temp.replace(path)
    return path


def run_evaluation(service: RAGService) -> EvalReport:
    """Run every golden case through the pipeline and score the answer."""
    config = service.config
    if not service.collection_ready():
        raise IndexNotReadyError(
            f"Qdrant collection '{config.collection_name}' does not exist, so there is nothing to evaluate."
        )

    dataset = load_eval_dataset(config)
    applicable = [
        case for case in dataset if case_is_applicable(case, service.settings)
    ]
    skipped_ids = [
        str(case["id"])
        for case in dataset
        if not case_is_applicable(case, service.settings)
    ]

    started_at = datetime.now(UTC)
    logger.info(
        "Started Evaluation",
        extra={
            "project": config.name,
            "cases": len(applicable),
            "skipped": len(skipped_ids),
        },
    )

    if skipped_ids:
        logger.warning(
            "Skipped cases whose required sources were not indexed.",
            extra={"skipped_case_ids": skipped_ids},
        )

    case_results = [_evaluate_case(service, case) for case in applicable]
    completed_at = datetime.now(UTC)
    passed = sum(1 for case in case_results if case.success)
    summary = EvalSummary(
        project=config.name,
        collection_name=config.collection_name,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=round((completed_at - started_at).total_seconds(), 2),
        dataset_size=len(case_results),
        passed_cases=passed,
        failed_cases=len(case_results) - passed,
        skipped_cases=len(skipped_ids),
        skipped_case_ids=skipped_ids,
        success_rate=round(passed / len(case_results), 4) if case_results else 0.0,
        success=passed == len(case_results),
    )

    result = EvalReport(summary=summary, cases=case_results)
    save_latest_eval_result(config, result)
    logger.info(
        "Evaluation finished",
        extra={
            "project": config.name,
            "passed": summary.passed_cases,
            "failed": summary.failed_cases,
            "skipped": summary.skipped_cases,
            "duration_seconds": summary.duration_seconds,
        },
    )
    return result


def _evaluate_case(service: RAGService, case: dict[str, Any]) -> EvalCaseResult:
    """Answer one golden case and score it against every metric."""
    from deepeval.test_case import LLMTestCase

    query_result = service.query(case["query"])
    test_case = LLMTestCase(
        input=case["query"],
        actual_output=query_result.response.answer,
        expected_output=case["expected_answer"],
        retrieval_context=query_result.retrieval_context,
    )
    metric_results = [_measure(metric, test_case) for metric in build_rag_metrics()]
    return EvalCaseResult(
        id=str(case["id"]),
        category=case.get("category"),
        query=case["query"],
        expected_answer=case["expected_answer"],
        actual_answer=query_result.response.answer,
        sources=[citation.label for citation in query_result.response.citations],
        retrieval_context=query_result.retrieval_context,
        metrics=metric_results,
        success=all(metric.success for metric in metric_results),
    )


def _measure(metric: Any, test_case: Any) -> EvalMetricResult:
    """Run one metric and capture its verdict."""
    threshold = _as_float(getattr(metric, "threshold", None))
    try:
        metric.measure(test_case, _show_indicator=False)
    except Exception as e:
        logger.warning("Evaluation metric raised", extra={"metric": metric})
        return EvalMetricResult(
            threshold=threshold,
            name=display_name(metric),
            score=None,
            success=False,
            reason=None,
            error=str(e),
        )
    return EvalMetricResult(
        threshold=threshold,
        name=display_name(metric),
        score=_as_float(getattr(metric, "score", None)),
        success=bool(getattr(metric, "success", False)),
        reason=getattr(metric, "reason", None),
        error=getattr(metric, "error", None),
    )


def display_name(metric: Any) -> str:
    """Return label of metric."""
    raw = getattr(metric, "name", None) or type(metric).__name__.removesuffix("Metric")
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", str(raw))
    return " ".join(spaced.split())


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None
