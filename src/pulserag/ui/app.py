"""Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from pulserag.core.logging import configure_logging
from pulserag.core.service import RAGService
from pulserag.core.settings import get_settings
from pulserag.ui.views import query, sources

__all__ = ["main"]


@st.cache_resource
def get_service() -> RAGService:
    """Build the process-wide service on first use; later calls return the same one."""
    settings = get_settings()
    configure_logging(settings.log_level)
    return RAGService.from_settings(settings)


def _render_status_strip(service: RAGService) -> None:
    """Show which project is served and whether it can answer questions."""
    problems = [check for check in service.readiness_check() if not check.healthy]

    columns = st.columns(3)
    columns[0].metric("Project", service.config.name)
    columns[1].metric("Status", "Not ready" if problems else "Ready")
    columns[2].metric("Collection", service.config.collection_name)

    if problems:
        st.warning(
            "**Not ready to answer questions.**\n\n"
            + "\n".join(f"- **{check.name}**: {check.details}" for check in problems)
        )


def main() -> None:
    """Configure the page and render every tab."""
    st.set_page_config(page_title="PulseRAG", page_icon="🩺", layout="wide")
    st.title("PulseRAG")
    st.caption(
        "Clinical guideline question answering, grounded in an indexed corpus of published guidance."
    )

    service = get_service()
    _render_status_strip(service)
    st.divider()

    query_tab, sources_tab = st.tabs(["Ask Questions", "Sources"])
    with query_tab:
        query.render(service)
    with sources_tab:
        sources.render(service)
