"""Managing the CORPUS an operator controls: Uploaded PDFs and PubMed Status."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pulserag.core.indexing import get_qdrant_client
from pulserag.core.project import ProjectConfig
from pulserag.core.settings import AppSettings

logger = logging.getLogger(__name__)
__all__ = []
_PDF_MAGIC = b"%PDF-"
_SCROLL_PAGE_SIZE = 256
_SCROLL_MAX_POINTS = 20_000
_PUBMED_PAYLOAD_FIELDS = [
    "source",
    "query",
    "ref_doc_id",
    "doc_id",
    "document_id",
    "URL",
    "Title of this paper",
    "title",
]


@dataclass(frozen=True)
class SourceFileRecord:
    """One Guideline PDF stored on the disk."""

    name: str
    size_bytes: int
    modified_at: datetime


@dataclass(frozen=True)
class PubMedQueryRecord:
    """How much indexed material one PubMed query account for."""

    query: str
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class PubMedStatus:
    """Config vs Actually indexed PubMed Status."""

    enabled: bool
    configured_queries: list[str]
    configured_query_limit: int
    configured_max_results: int
    indexed_query_summaries: list[PubMedQueryRecord]
    indexed_document_count: int
    indexed_chunk_count: int
    truncated: bool = False


class SourceManager:
    """Filesystem and index-inspection operations for one project's sources."""

    def __init__(self, config: ProjectConfig, settings: AppSettings) -> None:
        """Prepare the guidelines directory."""
        self.config = config
        self.settings = settings
        self.guideline_dir = config.guideline_dir
        self.guideline_dir.mkdir(parents=True, exist_ok=True)

    def list_sources(self) -> list[SourceFileRecord]:
        """List guideline PDFs, sorted by filename."""
        records: list[SourceFileRecord] = []
        for path in sorted(self.guideline_dir.glob("*.pdf")):
            try:
                records.append(self._describe(path))
            except OSError:
                logger.warning(
                    "Skipping unreadable source file.", extra={"file": path.name}
                )
        return records

    def save_source(self, filename: str, content: bytes) -> SourceFileRecord:
        """Validate and store an uploaded PDF."""
        path = self._resolve_source_path(filename)

        if not content:
            raise ValueError("Uploaded file is empty.")
        if len(content) > self.settings.max_upload_bytes:
            limit_mb = self.settings.max_upload_bytes / (1024 * 1024)
            raise ValueError(f"File exceeds the {limit_mb:.0f} MiB upload limit.")
        if not content.startswith(_PDF_MAGIC):
            raise ValueError("File does not look like a PDF (missing %PDF- signature).")

        path.write_bytes(content)
        logger.info(
            "Stored source file", extra={"file": path.name, "size_bytes": len(content)}
        )
        return self._describe(path)

    def delete_source(self, filename: str) -> None:
        """Delete a guideline PDF."""
        path = self._resolve_source_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Source '{path.name}' does not exist.")
        path.unlink()
        logger.info("Deleted source file", extra={"file": path.name})

    def pubmed_status(
        self,
        *,
        configured_queries: list[str],
        configured_query_limit: int,
        configured_max_results: int,
    ) -> PubMedStatus:
        """Report configured PubMed settings alongside what is really indexed."""
        enabled = (
            self.settings.pubmed_enabled
            and configured_query_limit > 0
            and configured_max_results > 0
        )
        base = PubMedStatus(
            enabled=enabled,
            configured_queries=configured_queries,
            configured_query_limit=configured_query_limit,
            configured_max_results=configured_max_results,
            indexed_query_summaries=[],
            indexed_document_count=0,
            indexed_chunk_count=0,
        )

        try:
            payloads, truncated = self._scroll_pubmed_payloads()
        except Exception:
            logger.debug(
                "Could not read PubMed status from the collection.",
                extra={"collection": self.config.collection_name},
                exc_info=True,
            )
            return base

        summaries = summarize_pubmed_payloads(payloads)
        return PubMedStatus(
            enabled=enabled,
            configured_queries=configured_queries,
            configured_query_limit=configured_query_limit,
            configured_max_results=configured_max_results,
            indexed_query_summaries=summaries,
            indexed_document_count=sum(summary.document_count for summary in summaries),
            indexed_chunk_count=sum(summary.chunk_count for summary in summaries),
            truncated=truncated,
        )

    def _scroll_pubmed_payloads(self) -> tuple[list[dict[str, Any]], bool]:
        """Page through PubMed points, returning their payloads and a truncation flag."""
        from qdrant_client.http import models as rest

        client = get_qdrant_client(self.settings)
        pubmed_only = rest.Filter(
            must=[
                rest.FieldCondition(key="source", match=rest.MatchValue(value="pubmed"))
            ]
        )

        payloads: list[dict[str, Any]] = []
        offset: Any = None
        truncated = False

        while True:
            points, offset = client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=pubmed_only,
                limit=_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=_PUBMED_PAYLOAD_FIELDS,
                with_vectors=False,
            )
            payloads.extend((point.payload or {}) for point in points)

            if offset is None:
                break
            if len(payloads) >= _SCROLL_MAX_POINTS:
                truncated = True
                logger.warning(
                    "PubMed status scan hit its point cap; counts are a lower bound.",
                    extra={"cap": _SCROLL_MAX_POINTS},
                )
                break

        return payloads, truncated

    def _describe(self, path: Path) -> SourceFileRecord:
        """Build a record from a file's current stat information."""
        stat = path.stat()
        return SourceFileRecord(
            name=path.name,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )

    def _resolve_source_path(self, filename: str) -> Path:
        """Map a client-supplied filename onto a safe path inside the data dir."""
        safe_name = Path(filename).name
        if not safe_name or safe_name != filename:
            raise ValueError(
                "Invalid source filename: expected a plain filename with no path."
            )
        if Path(safe_name).suffix.lower() != ".pdf":
            raise ValueError("Only PDF sources are supported.")
        return self.guideline_dir / safe_name


def summarize_pubmed_payloads(
    payloads: list[dict[str, Any]],
) -> list[PubMedQueryRecord]:
    """Group PubMed chunk payloads into per-query document and chunk counts."""
    grouped: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        if payload.get("source") != "pubmed":
            continue
        query = str(payload.get("query") or "Unspecified query")
        summary = grouped.setdefault(query, {"document_keys": set(), "chunk_count": 0})
        summary["chunk_count"] += 1
        summary["document_keys"].add(_pubmed_document_key(payload))

    return sorted(
        (
            PubMedQueryRecord(
                query=query,
                document_count=len(summary["document_keys"]),
                chunk_count=summary["chunk_count"],
            )
            for query, summary in grouped.items()
        ),
        key=lambda record: record.query,
    )


def _pubmed_document_key(payload: dict[str, Any]) -> str:
    """Derive a stable per-document identity from a chunk payload."""
    for key in (
        "ref_doc_id",
        "doc_id",
        "document_id",
        "URL",
        "Title of this paper",
        "title",
    ):
        value = payload.get(key)
        if value:
            return str(value)
    return "unknown-pubmed-document"
