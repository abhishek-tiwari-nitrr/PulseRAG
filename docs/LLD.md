# Low-Level Design: PulseRAG

---

## 1. Code map

```text
PulseRAG/
├── .python-version            Python 3.14 for uv
├── pyproject.toml             project name + libraries
├── .gitignore                 keeps .env, .venv and uploads out of Git
├── .env.example               template for keys
├── streamlit_app.py           Entry Point
└── src/pulserag/              FIRST BUILDING
    ├── __init__.py            PACKAGE_ROOT
    ├── core/
    │   ├── __init__.py        list of core modules
    │   ├── settings.py        AppSettings, get_settings()
    │   ├── logging.py         configure_logging()
    │   ├── exceptions.py      named errors
    │   ├── schemas.py         Citation, RAGResponse, ReadinessCheck
    │   ├── project.py         the plugin template
    │   ├── registry.py        name -> plugin lookup
    │   ├── indexing.py        Qdrant, build and load the index
    │   ├── retrieval.py       search + LLM settings
    │   ├── generation.py      citations, evidence, confidence
    │   └── service.py         RAGService
    ├── projects/pulserag/     SIDE BUILDING: the medical plugin
    │   ├── config.py          prompt, disclaimer, collection name
    │   ├── ingestor.py        loads PDF, PubMed, seed documents
    │   └── data/guidelines/WHO_BP.pdf
    └── ui/
        ├── app.py             page frame, status strip, tabs
        └── views/
            ├── query.py       Ask Questions tab
            └── sources.py     Sources tab (Rebuild button)
```

---

## 2. Data model

```mermaid
classDiagram
    class AppSettings {
        <<pydantic BaseSettings, frozen>>
        qdrant_url: str
        qdrant_api_key: SecretStr
        openai_api_key: SecretStr
        openai_model = "gpt-4o-mini"
        embedding_model = "text-embedding-3-small"
        embedding_dimensions = 512
        chunk_size = 512
        chunk_overlap = 100
        similarity_top_k = 10
        similarity_cutoff: float
        max_guideline_files = 3
        secret(value) str
    }
    class ProjectConfig {
        <<dataclass, frozen>>
        name
        collection_name
        system_prompt
        disclaimer
        data_dir: Path
        guidelines_dir() Path
    }
    class DocumentIngestor {
        <<abstract>>
        config: ProjectConfig
        settings: AppSettings
        load_and_parse()* List~Document~
        enrich_metadata(docs)* List~Document~
        ingest() List~Document~
    }
    class PulseRAGIngestor
    class ProjectDefinition {
        <<dataclass, frozen>>
        config: ProjectConfig
        ingestor_class: type~DocumentIngestor~
    }
    class RAGService {
        definition: ProjectDefinition
        settings: AppSettings
        _index: VectorStoreIndex
        _index_lock: Lock
        _rebuilding: bool
        from_settings(settings)$
        readiness_checks() List~ReadinessCheck~
        ensure_index_loaded()
        rebuild_index() tuple
        query(question) RAGResponse
    }
    class RAGResponse {
        answer: str
        evidence: str
        citations: List~Citation~
        confidence: str
        disclaimer: str
    }
    class Citation {
        label: str
        source_org: str
        score: float
    }
    class ReadinessCheck {
        name: str
        healthy: bool
        detail: str
    }

    DocumentIngestor <|-- PulseRAGIngestor
    ProjectDefinition --> ProjectConfig
    ProjectDefinition ..> DocumentIngestor
    RAGService --> ProjectDefinition
    RAGService --> AppSettings
    RAGService ..> RAGResponse
    RAGService ..> ReadinessCheck
    RAGResponse --> Citation
```
