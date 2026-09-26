"""Building and loading Qdrant vector index."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient

from pulserag.core.project import ProjectConfig
from pulserag.core.settings import AppSettings

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import Document

logger = logging.getLogger(__name__)
__all__ = [
    "build_embed_model",
    "build_index",
    "build_vector_store",
    "collection_exists",
    "collection_vector_size",
    "get_qdrant_client",
    "load_index",
]

_CLIENT_CACHE: dict[tuple[str, str | None, float], QdrantClient] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def get_qdrant_client(settings: AppSettings) -> QdrantClient:
    """Return a cached Qdrant client for the configred endpoint."""
    api_key = settings.secret(settings.qdrant_api_key)
    key = (settings.qdrant_url, api_key, settings.qdrant_timeout_seconds)

    cached = _CLIENT_CACHE.get(key)
    if cached is not None:
        return cached

    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(key)
        if cached is None:
            logger.info("Connecting to Qdrant", extra={"qdrant": settings.qdrant_url})
            cached = QdrantClient(
                url=settings.qdrant_url,
                api_key=api_key,
                timeout=settings.qdrant_timeout_seconds,
            )
            _CLIENT_CACHE[key] = cached

        return cached


def collection_exists(settings: AppSettings, config: ProjectConfig) -> bool:
    """Check whether the collection exists and is reachable."""
    client = get_qdrant_client(settings)
    try:
        client.get_collection(config.collection_name)
    except Exception:
        logger.debug(
            "Collection not available",
            extra={"collection": config.collection_name},
            exc_info=True,
        )
        return False
    return True


def collection_vector_size(config: ProjectConfig, settings: AppSettings) -> int | None:
    """Return dense vector width for collection which was created."""
    client = get_qdrant_client(settings)
    parmas = client.get_collection(config.collection_name).config.params.vectors

    if parmas is None:
        return None

    if isinstance(parmas, dict):
        sizes = {
            int(spec.size) for spec in parmas.values() if getattr(spec, "size", None)
        }
        return next(iter(sizes)) if len(sizes) == 1 else None
    return int(parmas.size)


def build_index(
    config: ProjectConfig, settings: AppSettings, documents: list[Document]
) -> VectorStoreIndex:
    """Chunk, embed and index document, replacing the existing collection.

    Steps:
    1. Drop the existing collection.
    2. Split Documents into overlapping chunks.
    3. Embed each chunk.
    4. Write vectors and payload metadta into new qdrant collection
    """
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    client = get_qdrant_client(settings)
    _drop_collection(client, config.collection_name)

    logger.info(
        "Building Index",
        extra={
            "colection": config.collection_name,
            "documents": len(documents),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
        },
    )

    storage_context = StorageContext(vector_stores=build_vector_store(config, settings))

    splitter = SentenceSplitter(
        chunk_overlap=settings.chunk_overlap, chunk_size=settings.chunk_size
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
        embed_model=build_embed_model(settings),
    )
    logger.info("Index Build Complete", extra={"collection", config.collection_name})
    return index


def build_embed_model(settings: AppSettings) -> Any:
    """Create OpenAI Embedding Model."""
    from llama_index.embeddings.openai import OpenAIEmbedding

    api_key = settings.secret(settings.openai_api_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. The chunk cannot be embedded.")
    return OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=api_key,
        dimensions=settings.embedding_dimensions,
        embed_batch_size=settings.embedding_batch_size,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def build_vector_store(config: ProjectConfig, settings: AppSettings) -> Any:
    """Create LlamaIndex vector store."""
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        client=get_qdrant_client(settings),
        collection_name=config.collection_name,
    )


def _drop_collection(client: QdrantClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(
            "Dropped existing collection", extra={"collection": collection_name}
        )
    except Exception:
        logger.warning(
            "Could not drop the collection; it may not be existing yet.",
            extra={"collection": collection_name},
            exc_info=True,
        )


def load_index(config: ProjectConfig, settings: AppSettings) -> VectorStoreIndex:
    """Attach to the existing collection so the query can run."""
    from llama_index.core import VectorStoreIndex

    logger.info("Loaded Index", extra={"collection": config.collection_name})
    return VectorStoreIndex.from_vector_store(
        embed_model=build_embed_model(settings),
        vector_store=build_vector_store(config, settings),
    )
