"""The registry and the clinical-guideline plugin."""

import pytest
from llama_index.core.schema import Document

from pulserag.core.exceptions import ProjectNotFoundError
from pulserag.core.registry import get_project_definition
from pulserag.core.service import RAGService
from pulserag.projects.pulserag.ingestor import PulseRAGIngestor


def test_the_registry_resolves_the_plugin_and_rejects_unknown_names(make_settings):
    definition = get_project_definition("pulserag", make_settings())

    assert definition.config.collection_name == "pulserag_collection_openai_small"
    assert definition.ingestor_class is PulseRAGIngestor
    with pytest.raises(ProjectNotFoundError, match="pulserag"):
        RAGService.from_settings(make_settings(active_project="typo"))


def test_seed_documents_are_the_whole_corpus_when_other_sources_are_off(config, make_settings):
    settings = make_settings(max_guideline_files=0, pubmed_enabled=False)

    documents = PulseRAGIngestor(config, settings).ingest()

    assert len(documents) == 4
    assert {doc.metadata["source_org"] for doc in documents} == {"Bootstrap"}


def test_documents_are_labelled_by_publisher(config, make_settings):
    docs = [
        Document(text="a", metadata={"source": "pubmed"}),
        Document(text="b", metadata={"source_file": "fda_label.pdf"}),
        Document(text="c", metadata={"source_file": "guideline.pdf"}),
    ]

    labelled = PulseRAGIngestor(config, make_settings()).enrich_metadata(docs)

    assert [doc.metadata["source_org"] for doc in labelled] == ["PubMed", "FDA", "WHO"]
