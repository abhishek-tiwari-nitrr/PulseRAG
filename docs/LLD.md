# Low-Level Design

---

## 1. Project Foundation

```
src/pulserag/
├── api/routers/
│   ├── health.py          health(service) | ready(service, response)
│   ├── query.py           query(payload, service, settings) - applies both guardrails
│   ├── sources.py         list_sources | upload_source (async) | delete_source | reindex_sources
│   ├── guardrails.py      guardrail_status | test_input_guardrail | test_output_guardrail
│   └── evaluations.py     run_eval | latest_eval
├── core/
│   ├── settings.py        AppSettings - pydantic-settings, frozen, read once per process
│   ├── indexing.py        get_qdrant_client(settings) - one pooled QdrantClient per endpoint
│   ├── service.py         RAGService - owns the loaded index
│   ├── sources.py         SourceManager - files on disk
│   ├── exceptions.py      PulseRAGError subclasses, each with .status_code and .detail
│   └── logging.py         configure_logging(level, log_format)
└── ui/
    ├── app.py             main() - page shell and tabs
    ├── client.py          ApiClient - HTTP client returning ApiResult
    └── views/*.py         render(client) - one per tab
```

