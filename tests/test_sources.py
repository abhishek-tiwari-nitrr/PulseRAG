"""Version 4: saving, listing and deleting guideline PDFs, and the PubMed status. No Qdrant needed."""

import pytest

from pulserag.core import sources
from pulserag.core.sources import SourceManager, summarize_pubmed_payloads

PDF = b"%PDF-1.4 a tiny test document"


def test_a_valid_pdf_is_saved_and_listed(config, make_settings):
    manager = SourceManager(config, make_settings())

    manager.save_source("guideline.pdf", PDF)

    assert [record.name for record in manager.list_sources()] == ["guideline.pdf"]
    assert (config.guideline_dir / "guideline.pdf").read_bytes() == PDF


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("notes.txt", PDF),
        ("../escape.pdf", PDF),
        ("fake.pdf", b"MZ not really a pdf"),
        ("empty.pdf", b""),
        ("big.pdf", PDF * 10),
    ],
)
def test_bad_uploads_are_rejected(config, make_settings, filename, content):
    manager = SourceManager(config, make_settings(max_upload_bytes=100))

    with pytest.raises(ValueError):
        manager.save_source(filename, content)
    assert manager.list_sources() == []


def test_delete_removes_the_file_and_reports_a_missing_one(config, make_settings):
    manager = SourceManager(config, make_settings())
    manager.save_source("guideline.pdf", PDF)

    manager.delete_source("guideline.pdf")

    assert manager.list_sources() == []
    with pytest.raises(FileNotFoundError):
        manager.delete_source("guideline.pdf")


def test_pubmed_chunks_are_counted_per_query_and_per_document():
    payloads = [
        {"source": "pubmed", "query": "diabetes", "ref_doc_id": "a"},
        {"source": "pubmed", "query": "diabetes", "ref_doc_id": "a"},
        {"source": "pubmed", "query": "diabetes", "ref_doc_id": "b"},
        {"source": "pubmed", "query": "hypertension", "ref_doc_id": "c"},
        {"source": "guideline_pdf", "query": None, "ref_doc_id": "d"},
    ]

    summaries = summarize_pubmed_payloads(payloads)

    assert [(s.query, s.document_count, s.chunk_count) for s in summaries] == [
        ("diabetes", 2, 3),
        ("hypertension", 1, 1),
    ]


def test_pubmed_status_still_shows_the_settings_when_qdrant_is_down(
    config, make_settings, monkeypatch
):
    def down(settings):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(sources, "get_qdrant_client", down)
    manager = SourceManager(
        config, make_settings(pubmed_query_limit=1, pubmed_max_results=5)
    )

    status = manager.pubmed_status(
        configured_queries=["diabetes"],
        configured_query_limit=1,
        configured_max_results=5,
    )

    assert status.enabled is True
    assert status.configured_queries == ["diabetes"]
    assert status.indexed_document_count == 0
