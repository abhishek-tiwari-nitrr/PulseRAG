"""The "Sources" tab: rebuild the index from every source."""

from __future__ import annotations

import streamlit as st

from pulserag.core.exceptions import PulseRAGError
from pulserag.core.service import RAGService
from pulserag.core.sources import PubMedStatus, SourceFileRecord, SourceManager
from pulserag.ui.formatting import format_size, format_timestamp

__all__ = ["render"]
_NOTICE_KEY = "sources_notice"


def render(service: RAGService) -> None:
    """Source management tab"""
    st.subheader("Corpus sources")
    st.caption(
        "Uploaded PDFs are parsed at index time. PubMed ingestion is controlled by environment settings and refreshed on each rebuild."
    )
    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        st.success(notice)

    manager = SourceManager(service.config, service.settings)
    _render_upload(service, manager)

    records = manager.list_sources()
    st.divider()
    _render_source_table(records)

    st.divider()
    _render_pubmed_status(
        manager.pubmed_status(
            configured_queries=list(getattr(service.definition.ingestor_class, "pubmed_queries", ())),
            configured_query_limit=service.settings.pubmed_query_limit,
            configured_max_results=service.settings.pubmed_max_results,
        )
    )

    st.divider()
    st.markdown("**Rebuild**")
    st.caption(
        "Re-ingests every source and replaces the collection. Required after any upload or delete, and after changing ingestion settings. Takes minutes."
    )
    if st.button("Rebuild index now", key="sources_rebuild"):
        _rebuild(service, "Rebuilt the collection.")


def _render_upload(service: RAGService, manager: SourceManager) -> None:
    """Upload one or more PDFs, then rebuild the index."""
    uploaded = st.file_uploader(
        "Add guideline PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Stored in the project's data directory and parsed on the next rebuild.",
        key="sources_uploader",
    )
    if not st.button("Upload and rebuild index", disabled=not uploaded, key="sources_upload"):
        return

    failures = []
    for file in uploaded or []:
        try:
            manager.save_source(file.name, file.getvalue())
        except ValueError as exc:
            failures.append(f"- {file.name}: {exc}")
    if failures:
        st.error("Some uploads failed:\n\n" + "\n".join(failures))

    _rebuild(service, "Uploaded sources and rebuilt the collection.")


def _render_source_table(records: list[SourceFileRecord]) -> None:
    """Show the PDFs currently on disk."""
    st.markdown("**Uploaded PDFs**")
    if not records:
        st.info("No PDFs uploaded yet. The index will use PubMed and bootstrap documents only.")
        return
    st.dataframe(
        [
            {
                "Name": record.name,
                "Size": format_size(record.size_bytes),
                "Updated": format_timestamp(record.modified_at),
            }
            for record in records
        ],
        width="stretch",
        hide_index=True,
    )



def _render_pubmed_status(pubmed: PubMedStatus) -> None:
    """Show configured PubMed settings next to what is really indexed."""
    st.markdown("**PubMed ingestion**")
    columns = st.columns(4)
    columns[0].metric("Status", "Enabled" if pubmed.enabled else "Disabled")
    columns[1].metric("Queries run", str(pubmed.configured_query_limit))
    columns[2].metric("Results per query", str(pubmed.configured_max_results))
    columns[3].metric("Abstracts indexed", str(pubmed.indexed_document_count))

    if pubmed.configured_queries:
        st.caption("Configured queries (only the first *Queries run* are executed):")
        for position, query in enumerate(pubmed.configured_queries, start=1):
            active = "▶" if position <= pubmed.configured_query_limit else "·"
            st.write(f"{active} {query}")

    if not pubmed.indexed_query_summaries:
        st.info(
            "No PubMed documents are present in the current index. If you expected some, "
            "rebuild the index."
        )
        return

    st.caption("Currently in the index:")
    st.dataframe(
        [
            {
                "Query": summary.query,
                "Abstracts": summary.document_count,
                "Chunks": summary.chunk_count,
            }
            for summary in pubmed.indexed_query_summaries
        ],
        width="stretch",
        hide_index=True,
    )
    if pubmed.truncated:
        st.caption("The collection is very large, so these counts are a lower bound.")


def _rebuild(service: RAGService, success_message: str) -> None:
    """Rebuild the index, then rerun the page with a success message."""
    with st.spinner("Rebuilding the index. This can take several minutes..."):
        try:
            documents, duration = service.rebuild_index()
        except PulseRAGError as exc:
            st.error(f"Rebuild failed: {exc.detail}")
            return
    st.session_state[_NOTICE_KEY] = (
        f"{success_message} Indexed {documents} document(s) in {duration:.1f}s."
    )
    st.rerun()