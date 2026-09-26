# Low-Level Design: PulseRAG

---

## 1. Code map

```text
streamlit_app.py
src/pulserag/
├── __init__.py           PACKAGE_ROOT
├── core/
│   ├── exceptions.py
│   ├── generation.py     Raw -> Structured Answer
│   ├── indexing.py       Qdrant Config
│   ├── logging.py
│   ├── project.py        Project Config
│   ├── registry.py       Get Project Definition
│   ├── retrieval.py      LLM call
│   ├── schemas.py        Data shapes
│   ├── service.py        Dashboard <-> Engine
│   ├── settings.py       App Settings
├── projects/pulserag/
│   ├── config.py         PulseRAG configuration
│   ├── ingestor.py       Loads source documents
│   ├── data/guidelines/  WHO_BP.pdf
└── ui/
    └── views/
tests/
```
