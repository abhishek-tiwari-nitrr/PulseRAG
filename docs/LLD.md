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
    │   ├── service.py         RAGService
    │   └── evaluation/
    │       ├── metrics.py
    │       └── runner.py
    ├── projects/pulserag/     SIDE BUILDING: the medical plugin
    │   ├── config.py          prompt, disclaimer, collection name
    │   ├── ingestor.py        loads PDF, PubMed, seed documents
    │   ├── data/guidelines    WHO_BP.pdf
    │   └── datasets/          golden_dataset.json
    └── ui/
        ├── app.py             page frame, status strip, tabs
        └── views/
            ├── query.py       Ask Questions tab
            ├── sources.py     Sources tab (Rebuild button)
            ├── formatting.py 
            └── evaluations.py Evaluations tab 
```

---

## 2. Data model

```mermaid
classDiagram
    class AppSettings {
        <<frozen, 24 fields>>
        qdrant_url: str
        openai_api_key: SecretStr
        secret(value) str
    }
    class ProjectConfig {
        name
        collection_name
        system_prompt
        disclaimer
        data_dir: Path
        golden_dataset_path: Path
    }
    class RAGService {
        from_settings(settings)$
        readiness_checks()
        ensure_index_loaded()
        rebuild_index() tuple
        query(question) QueryArtifacts
    }
    class RAGResponse {
        answer
        evidence
        citations: List~Citation~
        confidence
        disclaimer
    }
    class Citation {
        label
        source_org
        score
    }
    RAGService --> ProjectConfig
    RAGService --> AppSettings
    RAGResponse --> Citation
    class QueryArtifacts {
        response: RAGResponse
        retrieval_context: List~str~
    }
    class EvalReport {
        summary: EvalSummary
        cases: List~EvalCaseResult~
    }
    QueryArtifacts --> RAGResponse
```
