"""The service: readiness, querying and rebuilding, with Qdrant replaced by stubs."""

import pytest

from pulserag.core import service as service_module
from pulserag.core.exceptions import CorpusEmptyError, IndexNotReadyError
from pulserag.core.registry import ProjectDefinition
from pulserag.core.service import RAGService


class EmptyIngestor:
    def __init__(self, config, settings):
        pass

    def ingest(self):
        return []


class DownQdrant:
    def get_collections(self):
        raise ConnectionError("connection refused")


def make_service(config, settings):
    return RAGService(ProjectDefinition(config=config, ingestor_class=EmptyIngestor), settings)


def test_readiness_names_an_unreachable_qdrant(config, make_settings, monkeypatch):
    monkeypatch.setattr(service_module, "get_qdrant_client", lambda settings: DownQdrant())

    checks = make_service(config, make_settings()).readiness_check()

    assert [(c.name, c.healthy) for c in checks] == [
        ("qdrant", False),
        ("collection", False),
        ("generation", True),
    ]
    assert "connection refused" in checks[0].details


def test_asking_before_the_index_exists_is_a_readable_error(config, make_settings, monkeypatch):
    monkeypatch.setattr(service_module, "collection_exists", lambda config, settings: False)

    with pytest.raises(IndexNotReadyError, match="does not exist"):
        make_service(config, make_settings()).query("Anything?")


def test_an_empty_corpus_never_replaces_the_index(config, make_settings):
    service = make_service(config, make_settings())

    with pytest.raises(CorpusEmptyError):
        service.rebuild_index()
    assert service._index is None
