"""The "Evaluation" tab: run the golden-dataset suite and read the results."""

from __future__ import annotations

import streamlit as st

from pulserag.core.evaluation import load_latest_eval_result, run_evaluation
from pulserag.core.exceptions import PulseRAGError
from pulserag.core.schemas import EvalCaseResult, EvalReport
from pulserag.core.service import RAGService
from pulserag.ui.formatting import format_score, format_timestamp, status_icon

__all__ = ["render"]

_METRIC_HELP = {
    "Answer Relevance": "Does the answer address the question that was asked?",
    "Context Relevance": "Did retrieval fetch passages that are on-topic?",
    "Groundedness": "Is every claim in the answer supported by those passages?",
}


def render(service: RAGService) -> None:
    """Draw the evaluation tab."""
    st.subheader("Answer quality evaluation")
    st.caption(
        "Every question in the golden dataset is answered by the live pipeline and scored by three LLM judges."
    )

    if st.button("Run evaluation", type="primary", key="eval_run"):
        with st.spinner(
            "Running the golden-dataset evaluation. This can take minutes..."
        ):
            try:
                run_evaluation(service)
            except PulseRAGError as exc:
                st.error(f"Evaluation failed: {exc.detail}")
            else:
                st.success(
                    "Evaluation complete. The stored results below were updated."
                )

    report = load_latest_eval_result(service.config)
    if report is None:
        st.info(
            "No evaluation has been run yet. Use the button above to run the first one."
        )
        return
    _render_report(report)


def _render_report(report: EvalReport) -> None:
    """Render a full evaluation report."""
    summary = report.summary

    columns = st.columns(5)
    columns[0].metric("Result", "PASS" if summary.success else "FAIL")
    columns[1].metric("Cases", str(summary.dataset_size))
    columns[2].metric("Passed", str(summary.passed_cases))
    columns[3].metric("Failed", str(summary.failed_cases))
    columns[4].metric("Duration", f"{summary.duration_seconds:.1f}s")

    st.caption(
        f"Run completed {format_timestamp(summary.completed_at)} against "
        f"`{summary.collection_name}`. Success rate: {summary.success_rate * 100:.1f}%."
    )

    if summary.skipped_case_ids:
        st.info(
            f"**{len(summary.skipped_case_ids)} case(s) skipped** because a source they "
            f"require was not part of this index build: {', '.join(summary.skipped_case_ids)}. "
            "Index with the guideline PDF (MAX_GUIDELINE_FILES > 0) to run them."
        )

    st.markdown("**Cases**")
    st.dataframe(
        [
            {
                "Case": case.id,
                "Category": case.category or "",
                "Question": case.query,
                "Result": status_icon(case.success),
                "Metrics passed": (
                    f"{sum(1 for metric in case.metrics if metric.success)}/{len(case.metrics)}"
                ),
            }
            for case in report.cases
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Case detail**")
    for case in report.cases:
        with st.expander(
            f"{status_icon(case.success)} · {case.id} · {case.query}",
            expanded=not case.success,
        ):
            _render_case(case)


def _render_case(case: EvalCaseResult) -> None:
    """Render one case's answers, metrics and retrieved context."""
    left, right = st.columns(2)
    with left:
        st.caption("Expected answer")
        st.write(case.expected_answer)
    with right:
        st.caption("Actual answer")
        st.write(case.actual_answer)

    st.caption("Metric results")
    st.dataframe(
        [
            {
                "Metric": metric.name,
                "Measures": _METRIC_HELP.get(metric.name, ""),
                "Score": format_score(metric.score),
                "Threshold": format_score(metric.threshold),
                "Result": status_icon(metric.success),
                "Judge rationale": metric.reason or metric.error or "",
            }
            for metric in case.metrics
        ],
        width="stretch",
        hide_index=True,
    )

    st.caption("Sources cited")
    for source in case.sources or ["(none)"]:
        st.write(f"- {source}")

    with st.expander("Retrieved context the judges scored against"):
        for position, snippet in enumerate(case.retrieval_context, start=1):
            st.write(f"**{position}.** {snippet}")
        if not case.retrieval_context:
            st.write("(no context was retrieved)")
