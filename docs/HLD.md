# High-Level Design: PulseRAG
---

## 1. Purpose

PulseRAG answers questions about published clinical guidance using only an indexed corpus of source documents. Every answer comes with the passages behind it, a confidence grade derived from retrieval scores and a disclaimer. Safety guardrails check the question before any work is done and the drafted answer before it is shown. A golden-dataset evaluation measures whether the answers stay grounded as the corpus, prompt or model changes.

> Not a medical device. PulseRAG is an educational evidence assistant. It does not diagnose, prescribe, replace a clinician or guarantee that its corpus is complete or current.
---

## 2. System Context

```mermaid
flowchart LR
U(("User")) -->|HTTPS| ST

subgraph ST["Streamlit App"]
UI["Dashboard"] --> ENG["Engine"]
end

ENG -->|"vectors"| QC[("Qdrant Cloud")]
ENG -->|"embeddings, answers, judges"| OAI["OpenAI"]
UI -->|"input and output checks"| GRQ["Groq"]
ENG -->|"PDF parsing"| LP["LlamaCloud (LlamaParse)"]
ENG -->|"abstracts"| PM["PubMed"]

style ST fill:#e8f5e9,stroke:#2e7d32
```
---

## 3. Architecture

```mermaid
flowchart TB
    EP["streamlit_app.py"] --> UI

    subgraph UI["Presentation: pulserag.ui"]
        APP["app.py: page shell, <br/>cached RAGService"]
        Q["views/query.py"]
        S["views/sources.py"]
    end

    subgraph SV["Application: pulserag.core.service"]
        SVC["RAGService"]
    end

    subgraph EN["Engine: pulserag.core"]
        IDX["indexing"] 
        RET["retrieval"]
        GEN["generation"]
        REG["registry"]
    end

    subgraph FD["Foundations: pulserag.core"]
        SET["settings"] 
        LOG["logging"]
        EXC["exceptions"]
        SCH["schemas"]
        PRJ["project (plugin)"]
    end

    PLG["Plugin: pulserag.projects.pulserag<br/>config (prompt, disclaimer, collection) + ingestor"]

    APP --> SVC
    Q --> SVC
    S --> SVC
    SVC --> IDX & GEN & REG
    GEN --> RET
    REG -.->|"lazy import"| PLG
    PLG --> PRJ
    EN --> FD
```

---
## 4. Key flows

### 4.1 Index rebuild
```mermaid
sequenceDiagram
    actor Op as Operator
    participant SRC as Sources tab
    participant SVC as RAGService
    participant ING as PulseRAGIngestor
    participant IDX as indexing
    participant OAI as OpenAI
    participant QC as Qdrant Cloud

    Op->>SRC: Rebuild index now
    SRC->>SVC: rebuild_index()
    SVC->>SVC: acquire lock (non-blocking) or RebuildInProgressError
    SVC->>ING: ingest()
    ING-->>SVC: labelled documents
    alt no documents
        SVC-->>SRC: CorpusEmptyError (existing index kept)
    else documents
        SVC->>IDX: build_index(documents)
        IDX->>QC: delete collection
        IDX->>OAI: embed chunks (batches of 8)
        IDX->>QC: write vectors + metadata
        SVC->>SVC: swap in new index, release lock
        SVC-->>SRC: (documents, seconds)
        SRC->>SRC: store notice, st.rerun()
    end
```

### 4.2 Question answering

```mermaid
sequenceDiagram
    actor User
    participant ASK as Ask Questions tab
    participant SVC as RAGService
    participant OAI as OpenAI
    participant QC as Qdrant Cloud

    User->>ASK: question
    ASK->>SVC: query(question)
    SVC->>SVC: ensure_index_loaded()
    SVC->>OAI: embed question
    SVC->>QC: top-k (10) similarity search
    QC-->>SVC: scored chunks
    SVC->>OAI: one compact completion with the plugin system prompt
    OAI-->>SVC: answer
    SVC->>SVC: citations, evidence, confidence, disclaimer
    SVC-->>ASK: RAGResponse
    ASK-->>User: rendered answer
```

