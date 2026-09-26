"""The "Sources" tab: rebuild the index from every source."""

from __future__ import annotations

import streamlit as st

from pulserag.core.exceptions import PulseRAGError
from pulserag.core.service import RAGService

__all__ = ["render"]
_NOTICE_KEY = "sources_notice"


def render(service: RAGService) -> None:
    """Source management tab"""
    st.subheader("Corpus Sources")
    st.caption("Guideline pdfs in the data folder are parsed at index time.")
    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        st.success(notice)
    st.markdown("**Rebuild**")
    st.caption(
        "Re-ingests every source and replaces the collection. Required before the first question and after changing sources or ingestion settings. Takes few minutes."
    )

    if st.button("Rebuild Index Now", key="sources_rebuild"):
        _rebuild(service, "Rebuild the collection")


def _rebuild(service: RAGService, success_message: str) -> None:
    """Rebuild the index, then rerun the page."""
    with st.spinner("Rebuilding the index, This can serveral min..."):
        try:
            documents, duration = service.rebuild_index()
        except PulseRAGError as e:
            st.error(f"Rebuild Failed: {e}")
            return
    st.session_state[_NOTICE_KEY] = (
        f"{success_message} indexed {documents} docs in {duration:.1f}sec."
    )
    st.rerun()
