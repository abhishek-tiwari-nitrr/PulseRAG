"Ask Question Tab"

from __future__ import annotations

import streamlit as st

from pulserag.core.exceptions import PulseRAGError
from pulserag.core.schemas import RAGResponse
from pulserag.core.service import RAGService

__all__ = ["render"]
_CONFIDENCE_HELP = {
    "high": ("success", "Several closely matching passages supported this answer."),
    "moderate": (
        "info",
        "Some supporting material was found, but the match was partial.",
    ),
    "low": (
        "warning",
        "Little supporting material was found. Treat this answer with particular caution and check the sources directly.",
    ),
}
_PLACEHOLDER = (
    "What do the guidelines say about first-line treatment for type 2 diabetes?"
)


def render(service: RAGService) -> None:
    "Question Answer tab"
    st.subheader("Ask a clinical question")
    st.caption(
        "Answer are drawn only from the indexed corpus. If the corpus does not cover your question, the assistant should say so rather than guess."
    )

    question = st.text_area(
        "Question", placeholder=_PLACEHOLDER, key="query_question"
    ).strip()
    if not st.button("Ask", type="primary", key="query_submit"):
        return
    if not question:
        st.warning("Enter a question first")
        return
    with st.spinner("Searching the knowledge abs e and drafting an answer..."):
        try:
            response = service.query(question=question)
        except PulseRAGError as e:
            st.error(e.detail)
            return
    _render_answer(response=response)


def _render_answer(response: RAGResponse) -> None:
    """Render successful answer."""
    st.subheader("Answer")
    st.write(response.answer)
    callout, explanation = _CONFIDENCE_HELP[response.confidence]
    getattr(st, callout)(f"Confidence: **{response.confidence}** - {explanation}")

    st.subheader("Source")
    if response.citations:
        st.dataframe(
            [
                {
                    "Source": citation.label,
                    "Org": citation.source_org,
                    "Relevance": (
                        "-" if citation.score is None else f"{citation.score:.3f}"
                    ),
                }
                for citation in response.citations
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("(none retrieved)")

    with st.expander("Supporting Evidence (excerpts from retrieved passages)"):
        st.write(response.evidence)

    st.info(response.disclaimer)
