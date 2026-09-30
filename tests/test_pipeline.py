"""Indexing, retrieval and answer shaping, with a stub index instead of Qdrant and OpenAI."""

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode

from pulserag.core.generation import answer_question
from pulserag.core.indexing import build_embed_model
from pulserag.core.retrieval import build_query_engine


class StubIndex:
    """Stands in for a loaded VectorStoreIndex: records the engine settings, returns fixed nodes."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.engine_kwargs = {}

    def as_query_engine(self, **kwargs):
        self.engine_kwargs = kwargs
        return self

    def query(self, question):
        return Response(response="Metformin is usually first-line.", source_nodes=self.nodes)


def node(score, **metadata):
    return NodeWithScore(node=TextNode(text="Guidelines describe metformin.", metadata=metadata),
                         score=score)


def test_the_embedding_model_needs_a_key_and_uses_the_configured_width(make_settings):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_embed_model(make_settings(openai_api_key=None))

    assert build_embed_model(make_settings(embedding_dimensions=768)).dimensions == 768


def test_the_query_engine_gets_top_k_and_the_optional_cutoff(config, make_settings):
    index = StubIndex([])
    build_query_engine(index, config, make_settings(similarity_top_k=5))
    assert index.engine_kwargs["similarity_top_k"] == 5
    assert "node_post_processors" not in index.engine_kwargs

    build_query_engine(index, config, make_settings(similarity_cutoff=0.3))
    assert len(index.engine_kwargs["node_postprocessors"]) == 1


def test_an_answer_carries_ranked_citations_confidence_and_the_disclaimer(config, make_settings):
    index = StubIndex([
        node(0.62, source_org="WHO", source_file="WHO_BP.pdf", page_label="12"),
        node(0.58, source_org="WHO", source_file="WHO_BP.pdf", page_label="12"),
        node(0.40, source_org="PubMed", title="A trial"),
    ])

    response = answer_question(index, "First-line therapy?", config, make_settings())

    assert response.answer == "Metformin is usually first-line."
    assert [c.label for c in response.citations] == [
        "WHO: WHO_BP.pdf (page 12)",
        "PubMed: A trial",
    ]
    assert response.citations[0].score == pytest.approx(0.62)
    assert response.confidence == "high"
    assert response.disclaimer == config.disclaimer


def test_nothing_retrieved_means_low_confidence(config, make_settings):
    response = answer_question(StubIndex([]), "Anything?", config, make_settings())

    assert response.confidence == "low"
    assert response.citations == []
    assert "No Supporting Evidence" in response.evidence


def test_the_query_engine_gets_top_k_and_the_optional_cutoff(config, make_settings):
    index = StubIndex([])
    build_query_engine(index, config, make_settings(similarity_top_k=5, similarity_cutoff=None))
    assert index.engine_kwargs["similarity_top_k"] == 5
    assert "node_postprocessors" not in index.engine_kwargs

    build_query_engine(index, config, make_settings(similarity_cutoff=0.3))
    assert len(index.engine_kwargs["node_postprocessors"]) == 1


def test_an_answer_carries_ranked_citations_confidence_and_the_disclaimer(config, make_settings):
    index = StubIndex([
        node(0.62, source_org="WHO", source_file="WHO_BP.pdf", page_label="12"),
        node(0.58, source_org="WHO", source_file="WHO_BP.pdf", page_label="12"),
        node(0.40, source_org="PubMed", title="A trial"),
    ])

    result = answer_question(index, "First-line therapy?", config, make_settings())
    response = result.response

    assert response.answer == "Metformin is usually first-line."
    assert [c.label for c in response.citations] == [
        "WHO: WHO_BP.pdf (page 12)",  # the two WHO chunks become one citation
        "PubMed: A trial",
    ]
    assert response.citations[0].score == pytest.approx(0.62)
    assert response.confidence == "high"
    assert response.disclaimer == config.disclaimer
    assert result.retrieval_context == ["Guidelines describe metformin."] * 3  # what the judges read


def test_nothing_retrieved_means_low_confidence(config, make_settings):
    result = answer_question(StubIndex([]), "Anything?", config, make_settings())

    assert result.response.confidence == "low"
    assert result.response.citations == []
    assert "No Supporting Evidence" in result.response.evidence
    assert result.retrieval_context == []

