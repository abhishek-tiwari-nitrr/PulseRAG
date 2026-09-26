"""Expected, explainable failures."""

from __future__ import annotations

__all__ = [
    "CorpusEmptyError",
    "GenerationUnavailableError",
    "IndexNotReadyError",
    "ProjectNotFoundError",
    "PulseRAGError",
    "RebuildInProgressError",
]


class PulseRAGError(Exception):
    """Base class for every expected explainable application failure."""

    def __init__(self, details: str) -> None:
        super().__init__(details)
        self.detail = details


class ProjectNotFoundError(PulseRAGError):
    """Active Project names a plugin that is not registered/present."""


class IndexNotReadyError(PulseRAGError):
    """Qdrant Collection does not exists."""


class RebuildInProgressError(PulseRAGError):
    """Rebuild is running and holds use of collection."""


class CorpusEmptyError(PulseRAGError):
    """Ingestion produced no document so there is nothing to index."""


class GenerationUnavailableError(PulseRAGError):
    """Generation Provider is not configured or not reachable."""
