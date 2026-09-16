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

    PLUG["Project Plugin<br/>Prompts | Safety | Loaders"]

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

---

## 3. Design Overview

Running the PulseRAG dashboard and RAG application inside a single Streamlit process, while keeping external services such as Qdrant, OpenAI and Groq hosted separately.


```mermaid
flowchart LR

    B(("User Browser"))

    SC["Streamlit Cloud<br/>HTTPS"]

    subgraph SP["Single Streamlit Process"]
        direction TB
        UI["Dashboard Views"]
        CLIENT["In-Process Client"]
        H["Route Handlers<br/>Guardrails"]
        SVC["RAGService + RAG Engine"]

        UI --> CLIENT
        CLIENT --> H
        H --> SVC
    end

    Q[("Qdrant Cloud<br/>Vector Database")]
    OAI["OpenAI<br/>LLM"]
    GRQ["Groq<br/>Guardrails"]

    ENTRY["streamlit_app.py<br/>Entry Point"]

    B -->|"HTTPS"| SC
    SC --> UI

    SVC -->|"QDRANT_URL + API Key"| Q
    SVC --> OAI
    H --> GRQ

    ENTRY -.->|"Starts & prepares"| SP

    style SP fill:#e8f5e9,stroke:#2e7d32
    style Q fill:#fff3e0,stroke:#ef6c00
```


---

## 4. Guardrail Boundary

Guardrails are enforced in the API route handlers and only there. 

```mermaid
flowchart TB
    subgraph Guarded["Guarded - passes through api/routers/query.py"]
        HTTP["HTTP request"]
        INP["InProcessClient.query()"]
    end

    subgraph Unguarded["Unguarded - calls RAGService directly"]
        CLI["pulserag query"]
        EVAL["evaluation harness"]
    end

    HTTP --> Q["query.query(payload, service, settings)"]
    INP --> Q
    Q --> IG["input guard"] --> SVC["RAGService.query()"] --> OG["output guard"]
    CLI --> SVC
    EVAL --> SVC

    style Guarded fill:#e8f5e9,stroke:#2e7d32
    style Unguarded fill:#fff3e0,stroke:#ef6c00
```

- **Guardrails fail open.** If the Groq classifier is unreachable, the request proceeds unchecked and a warning is logged a deliberate availability trade-off.

---

## 5. Where the Data Lives

The Streamlit container uses **ephemeral storage** (means: files written to the container's local disk can disappear when the application restarts or is redeployed). 

```mermaid
flowchart LR

    subgraph Laptop["Engineer's Machine"]
        SRC["Source PDFs<br/>PubMed<br/>Bootstrap Docs"]
        INDEX["pulserag index<br/>Chunk | Embed"]
        SRC --> INDEX
    end

    INDEX -->|"Vectors"| QC[("Qdrant Cloud<br/>Persistent")]

    subgraph Cloud["Streamlit Container<br/>Ephemeral"]
        APP["Dashboard + RAG Engine"]
        UP["Uploaded PDFs"]
        EV["Evaluation Reports"]
    end

    QC -->|"Read at query time"| APP

    PKG["Package Data<br/>Golden Dataset | Seed PDF"]
    PKG -->|"Included in repository"| APP

    style QC fill:#e8f5e9,stroke:#2e7d32
    style UP fill:#ffebee,stroke:#c62828
    style EV fill:#ffebee,stroke:#c62828
```

| State| Where | Survives restart? |
| --- | --- | ---- | 
| **Vectors**| Qdrant Cloud | Yes |
| **Golden dataset, seed PDF**  | Inside the package | Yes |
| **Uploaded PDFs** | Container disk | **No** |
| **Evaluation reports** | Container disk | **No** |

---

## 6. Request flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant ST as Streamlit script
    participant IPC as InProcessClient
    participant H as query.query()
    participant GR as Guardrails (Groq)
    participant SVC as RAGService
    participant QC as Qdrant Cloud
    participant LLM as OpenAI

    User->>ST: asks a question (page reruns)
    ST->>IPC: client.query(question)
    IPC->>H: query.query(QueryRequest, service, settings)
    H->>GR: check_prompt_injection()
    alt blocked
        GR-->>H: blocked
        H-->>IPC: raise HTTPException(422)
        IPC-->>ST: ApiResult(error, status_code=422)
        ST-->>User: "blocked by a safety guardrail"
    else allowed
        H->>SVC: query()
        SVC->>QC: vector search over HTTPS
        QC-->>SVC: scored passages
        SVC->>LLM: one compact call
        LLM-->>SVC: draft
        H->>GR: check_safeguard_policy()
        alt violation
            H-->>IPC: raise HTTPException(422)
            IPC-->>ST: ApiResult(error, 422) - draft withheld
        else allowed
            H-->>IPC: RAGResponse
            IPC-->>ST: ApiResult(payload=model_dump(), 200)
            ST-->>User: answer | citations | confidence | disclaimer
        end
    end
```

---

# 7. Security Boundary and Access Control

The single-process deployment does **not** use a reverse proxy or a separate API server. The entire PulseRAG application runs behind the public HTTPS endpoint provided by Streamlit Cloud. This makes the Streamlit application's access control the **primary external security boundary**.

```mermaid
flowchart TB

    NET(("Internet"))
    SCP["Streamlit Cloud<br/>HTTPS Front Door"]
    PAGE["PulseRAG Dashboard"]
    ENG["RAG Engine<br/>+ API Credentials"]
    OAI["OpenAI<br/>API Usage"]

    NET --> SCP
    SCP -->|"Authenticated viewers only"| PAGE
    PAGE --> ENG
    ENG --> OAI

    style SCP fill:#e8f5e9,stroke:#2e7d32
    style PAGE fill:#e3f2fd,stroke:#1565c0
    style OAI fill:#ffebee,stroke:#c62828
```