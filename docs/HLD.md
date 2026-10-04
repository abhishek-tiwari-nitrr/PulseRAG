# High-Level Design: PulseRAG
---

## 1. Purpose

PulseRAG answers questions about published clinical guidance using only an indexed corpus of source documents. Every answer comes with the passages behind it, a confidence grade derived from retrieval scores and a disclaimer. Safety guardrails check the question before any work is done and the drafted answer before it is shown. A golden-dataset evaluation measures whether the answers stay grounded as the corpus, prompt or model changes.

> Not a medical device. PulseRAG is an educational evidence assistant. It does not diagnose, prescribe, replace a clinician or guarantee that its corpus is complete or current.
---

## 2. System Context

```mermaid
flowchart LR
    U(("Users")) --> ST
    subgraph ST["Streamlit app: one process"]
        UI["Dashboard"] --> SVC["RAGService"]
        UI --> EV["Evaluation runner"]
        EV --> SVC
        UI --> GRD["Guardrails"]
    end
    SVC --> QC[("Qdrant Cloud")]
    SVC --> OAI["OpenAI: embeddings, gpt-4o-mini"]
    SVC --> LP["LlamaParse"]
    SVC --> PM["PubMed"]
    EV --> JDG["OpenAI: gpt-4o judges (DeepEval)"]
    GRD --> GROQ["Groq: Prompt Guard, safeguard model"]
```

---

## 3. Architecture

```mermaid
flowchart TB
    EP["streamlit_app.py"] --> APP
    subgraph UIL["pulserag.ui"]
        APP["app.py"]
        VQ["views/query.py"]
        VS["views/sources.py"]
        VE["views/evaluations.py"]
        FMT["formatting.py"]
        VG["views/guardrails.py"]
    end
    subgraph CORE["pulserag.core"]
        SVC["service"]
        IDX["indexing"]
        RET["retrieval"]
        GEN["generation"]
        REG["registry"]
        FND["settings, logging, exceptions, schemas, project"]
        EVL["evaluation: metrics, runner"]
        GRL["guardrails"]
        SRM["sources"]
    end
    PLG["projects.pulserag: config, ingestor, golden dataset, safety policy"]
    APP --> VQ
    APP --> VS
    VQ --> SVC
    VS --> SVC
    SVC --> IDX
    SVC --> GEN
    GEN --> RET
    SVC --> REG
    REG -.->|"lazy"| PLG
    APP --> VE
    VE --> EVL
    EVL --> SVC
    APP --> VG
    VQ --> GRL
    VG --> GRL
    VS --> SRM
    SRM --> IDX
    style PLG fill:#fff3e0,stroke:#ef6c00
```

---
## 4. Key flows

### 4.1 Index Rebuild

```mermaid
sequenceDiagram
    actor Op as Operator
    participant SRC as Sources tab
    participant SVC as RAGService
    participant ING as PulseRAGIngestor
    participant IDX as indexing
    participant QC as Qdrant Cloud
    Op->>SRC: Rebuild / Upload / Delete
    SRC->>SVC: rebuild_index()
    SVC->>SVC: lock (non-blocking) or RebuildInProgressError
    SVC->>ING: ingest() PDFs, PubMed, seed documents
    alt no documents
        SVC-->>SRC: CorpusEmptyError (old index kept)
    else documents
        SVC->>IDX: build_index()
        IDX->>QC: drop collection
        IDX->>IDX: chunk (1024/100), embed with OpenAI
        IDX->>QC: write vectors + metadata
        IDX->>QC: create keyword indexes (source, query)
        SVC->>SVC: swap in new index, unlock
        SRC->>SRC: notice in session_state, st.rerun()
    end
```

### 4.2 Question Answering

```mermaid
sequenceDiagram
    actor User
    participant ASK as Ask tab
    participant PG as Groq Prompt Guard
    participant SVC as RAGService
    participant SG as Groq safeguard
    User->>ASK: question
    ASK->>PG: classify
    alt score >= threshold
        ASK-->>User: blocked at input
    else allowed or guard unavailable
        ASK->>SVC: query()
        SVC-->>ASK: QueryArtifacts
        ASK->>SG: policy + question + draft
        alt violation
            ASK-->>User: blocked at output (draft withheld)
        else allowed
            ASK-->>User: answer, confidence, citations, disclaimer
        end
    end
```

### 4.3 Evaluation Run

```mermaid
sequenceDiagram
    actor Op as Operator
    participant TAB as Evaluation tab
    participant RUN as runner
    participant SVC as RAGService
    participant DE as DeepEval judges
    Op->>TAB: Run evaluation
    TAB->>RUN: run_evaluation(service)
    RUN->>SVC: collection_ready() or IndexNotReadyError
    RUN->>RUN: load dataset, skip cases whose sources were not indexed
    loop applicable cases
        RUN->>SVC: query(case)
        RUN->>DE: 3 metrics
    end
    RUN->>RUN: atomic write data/evals/pulserag_latest.json
    TAB-->>Op: report
```

### 4.4 The guardrail boundary

```mermaid
flowchart TB
    subgraph G["Guarded"]
        ASK["Ask tab"]
    end
    subgraph NG["Unguarded by design"]
        EVAL["Evaluation runner"]
    end
    subgraph P["Probes"]
        GT["Guardrails tab"]
    end
    ASK --> IG["input guard"] --> SVC["RAGService.query"] --> OG["output guard"]
    EVAL --> SVC
    GT --> IG
    GT --> OG
```

