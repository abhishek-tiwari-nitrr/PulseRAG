# High-Level Design

---

## 1. Purpose

PulseRAG answers questions about published clinical guidance using only an indexed corpus of source documents. Every answer returns the passages behind it, a confidence grade derived from retrieval scores and a disclaimer. Safety guardrails check the question before any work is done and the drafted answer before it is shown.

---

## 2. Platform Architecture

PulseRAG is designed as a **layered application**.

### Architecture Overview

```mermaid
flowchart TB

    subgraph INTERFACES["Interfaces"]
        UI["Streamlit Dashboard<br/>pulserag.ui"]
        API["FastAPI API<br/>pulserag.api"]
        CLI["CLI<br/>pulserag.cli"]
    end

    SVC["RAGService<br/>Single Stateful Service"]

    subgraph CORE["Core RAG Engine"]
        IDX["Indexing"]
        RET["Retrieval"]
        GEN["Generation"]
        GRD["Guardrails"]
        EVAL["Evaluation"]
    end

    PLUG["Project Plugin<br/>Prompts · Safety · Loaders"]

    UI -->|"client.query()"| API
    API -->|"validated request"| SVC
    CLI --> SVC

    API --> GRD

    SVC --> IDX
    SVC --> RET
    RET --> GEN
    SVC --> EVAL
    SVC --> PLUG

    style INTERFACES fill:#e3f2fd,stroke:#1565c0
    style CORE fill:#e8f5e9,stroke:#2e7d32
    style PLUG fill:#fff3e0,stroke:#ef6c00
```