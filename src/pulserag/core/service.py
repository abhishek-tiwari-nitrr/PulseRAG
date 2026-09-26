"""The application service: one object that owns the RAG pipeline's state."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pulserag.core.exceptions import (
    CorpusEmptyError,
    GenerationUnavailableError,
    IndexNotReadyError,
    ProjectNotFoundError,
    RebuildInProgressError,
)
from pulserag.core.generation import answer_question
from pulserag.core.indexing import (
    build_index,
    collection_exists,
    collection_vector_size,
    get_qdrant_client,
    load_index,
)
from pulserag.core.project import ProjectConfig
from pulserag.core.registry import ProjectDefinition, get_project_definition
from pulserag.core.schemas import RAGResponse, ReadinessCheck
from pulserag.core.settings import AppSettings

logger = logging.getLogger(__name__)
__all__ = ["RAGService"]


class RAGService:
    """Owns the loaded index and expose the operation built on top of it."""

    def __init__(self, definition: ProjectDefinition, settings=AppSettings):
        self.definition = definition
        self.settings = settings
        self._index: Any | None = None
        self._index_lock = threading.Lock()
        self._rebuilding = False

    @classmethod
    def from_settings(cls, settings: AppSettings) -> RAGResponse:
        """Build a service for the project named by ACTIVE_PROJECT."""
        try:
            definition = get_project_definition(settings.active_project, settings)
        except ValueError as e:
            raise ProjectNotFoundError(str(e)) from e
        return cls(definition=definition, settings=settings)

    # introspection
    @property
    def config(self) -> ProjectConfig:
        """Configuration of the project this service serves."""
        return self.definition.config

    def collection_ready(self) -> bool:
        """Whether the project's collection exists and can be described."""
        return collection_exists(settings=self.settings, config=self.config)

    def readiness_check(self) -> list[ReadinessCheck]:
        """
        Probe every dependency needed to answer a question.

        Four checks to check for failure:
        1. Qdrant - is vector store reachable ?
        2. collection - does project collection exists ?
        3. Embedding Dimensions - collection width matches EMBEDDING_DIMENSION ?
        4. Generation - is openai key configured ?
        """
        checks: list[ReadinessCheck] = []
        try:
            get_qdrant_client(self.settings).get_collections()
            checks.append(
                ReadinessCheck(name="qdrant", healthy=True, details="reachable")
            )
            qdrant_up = True
        except Exception as e:
            checks.append(
                ReadinessCheck(
                    name="qdrant",
                    healthy=False,
                    details=f"{self.settings.qdrant_url} unreachable: {e}",
                )
            )
            qdrant_up = False

        collection_ok = qdrant_up and self.collection_ready()
        checks.append(
            ReadinessCheck(
                name="collection",
                healthy=collection_ok,
                details=(
                    f"{self.config.collection_name} present"
                    if collection_ok
                    else f"{self.config.collection_name} has not been built yet"
                ),
            )
        )

        if collection_ok:
            checks.append(self._embedding_dimension_check())

        generation_ok = self.settings.openai_api_key is not None

        checks.append(
            ReadinessCheck(
                name="generation",
                healthy=generation_ok,
                details=(
                    f"model {self.settings.openai_model} configured"
                    if generation_ok
                    else "OPEN_API_KEY not configured; question cannot be answered"
                ),
            )
        )
        return checks

    def _embedding_dimension_check(self) -> ReadinessCheck:
        """Compare the live collection vector width with configuration."""
        expected = self.settings.embedding_dimensions
        try:
            actual = collection_vector_size(config=self.config, settings=self.settings)
        except Exception as e:
            return ReadinessCheck(
                name="embedding_dimensions",
                healthy=False,
                details=f"Could not read collection vector size: {e}",
            )

        if actual is None:
            return ReadinessCheck(
                name="embedding_dimensions",
                healthy=True,
                detail="collection uses multiple or unnamed vectors; size not verified",
            )
        if actual != expected:
            return ReadinessCheck(
                name="embedding_dimensions",
                healthy=False,
                details=f"Collection stores {actual}-dim vector but EMBEDDING_DIMENSIONS is {expected}-dim. Rebuild the Index.",
            )
        return ReadinessCheck(
            name="embedding_dimensions", healthy=True, details=f"{actual}-dim"
        )

    # index lifecycle
    def ensure_index_loaded(self) -> Any:
        """Return the loaded index, loading it once if necessary."""
        index = self._index
        if index is not None:
            return index

        if self._rebuilding:
            raise RebuildInProgressError(
                "The Index is being rebuild. Retry after some time."
            )

        with self._index_lock:
            if self._index is not None:
                return self._index
            if not self.collection_ready():
                raise IndexNotReadyError(
                    f"Qdrant Collection {self.config.collection_name} does not exists. Build the index first."
                )

            self._index = load_index(config=self.config, settings=self.settings)
            return self._index

    def rebuild_index(self) -> tuple[int, float]:
        """Re-ingest every source and rebuild the collection from Scratch."""
        if self._index_lock.acquire(blocking=False):
            raise RebuildInProgressError("Another index rebuild is already running")
        started = time.perf_counter()
        try:
            self._rebuilding = True
            logger.info("Started index rebuild", extra={"project": self.config.name})
            ingestor = self.definition.ingestor_class(
                config=self.config, settings=self.settings
            )
            documents = ingestor.ingest()

            if not documents:
                raise CorpusEmptyError("Ingestion produced no documents.")
            new_index = build_index(
                config=self.config, settings=self.settings, documents=documents
            )
            self._index = new_index
            duration = round(time.perf_counter - started, 2)
            logger.info(
                "Index rebuild finished",
                extra={
                    "project": self.config.name,
                    "documents": len(documents),
                    "duration_seconds": duration,
                },
            )
            return len(documents), duration
        finally:
            self._rebuilding = False
            self._index_lock.release()

    # query
    def query(self, question: str) -> RAGResponse:
        """Answer a question from indexed corpus."""
        index = self.ensure_index_loaded()
        try:
            return answer_question(
                index=index,
                question=question,
                config=self.config,
                settings=self.settings,
            )
        except RuntimeError as e:
            if "OPENAI_API_KEY" in str(e):
                raise GenerationUnavailableError(str(e)) from e
            raise
