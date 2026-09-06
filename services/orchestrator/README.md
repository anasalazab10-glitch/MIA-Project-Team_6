# LEDGER Orchestrator Service (`orchestrator-api`)

The **Orchestrator Service** is the central nervous system of **Project LEDGER (Financial Document Intelligence Agent)**. It coordinates all six downstream microservices: Document Processing, Hybrid Retrieval, Reasoning Brain (LangGraph), Answer Validation, Gradio UI, and Evaluation.

---

## Key Responsibilities

1. **Query Orchestration (`POST /run`)**:
   - Accepts natural language questions from the Gradio UI or Evaluation benchmark.
   - Dispatches questions to the **Reasoning Service** (`agent-service`).
   - Sends candidate answers to the **Answer Validator** (`answer-validator-api`).
   - **Critical Safety Guard**: If an answer fails strict schema or citation validation, it is intercepted and converted into a compliant `insufficient_evidence` answer to prevent any hallucinated figures from reaching the user.
   - Computes request latency and logs queries for the corpus dashboard and Langfuse observability.

2. **Document Ingestion (`POST /ingest`)**:
   - Accepts raw financial report PDFs (TAT-DQA compliant).
   - Routes raw PDFs to **Document Processing** (`doc-processor-api`) for OCR, layout parsing, and table extraction.
   - Forwards structured elements to **Retrieval** (`retrieval-api`) for chunking and vector indexing in Qdrant and BM25.
   - Persists document summaries and detected tables in the catalog.

3. **Dashboard & Catalog APIs**:
   - `GET /dashboard/stats`: Returns corpus-level statistics (total documents, table counts, recent queries with latencies) for the Gradio UI dashboard.
   - `GET /documents`: Lists all indexed documents.
   - `GET /documents/{document_id}`: Returns document details, preview elements, and extracted tables.

4. **Health Check (`GET /health`)**:
   - Probes downstream services (`doc-processor`, `retrieval`, `reasoning`, `validator`) and reports status.

---

## API Endpoints

### 1. Execute Query
- **Endpoint**: `POST /run`
- **Request**:
  ```json
  {
    "question": "What was the operating income reported in 2020?",
    "session_id": "session_123",
    "document_id": "optional_doc_filter"
  }
  ```
- **Response (Strict Answer Schema)**:
  ```json
  {
    "answer_type": "direct",
    "evidence": [
      { "document_id": "doc_017", "page": 1, "section": "Income Statement" }
    ],
    "params": {
      "value": "$142.5M"
    },
    "validation_status": "valid",
    "latency_ms": 142.5
  }
  ```

### 2. Ingest PDF Document
- **Endpoint**: `POST /ingest`
- **Format**: `multipart/form-data`
- **Fields**:
  - `file`: Raw `.pdf` file (binary)
  - `document_id`: (optional) custom ID
  - `dpi`: (optional, default: 200)

### 3. Corpus Dashboard Stats
- **Endpoint**: `GET /dashboard/stats`
- **Response**:
  ```json
  {
    "total_documents": 5,
    "total_tables": 12,
    "total_elements": 148,
    "documents": [...],
    "tables": [...],
    "recent_queries": [
      {
        "query_id": "a1b2c3d4",
        "question": "...",
        "answer_type": "direct",
        "validation_passed": true,
        "latency_ms": 320.5,
        "timestamp": "2026-09-06T04:20:00Z"
      }
    ]
  }
  ```

---

## Configuration

Environment variables (with sensible defaults):

| Variable | Default (Local) | Default (Docker) | Description |
|---|---|---|---|
| `DOC_PROCESSOR_URL` | `http://localhost:8002` | `http://doc-processor:8000` | Document processing service |
| `RETRIEVAL_URL` | `http://localhost:8000` | `http://dense-retrieval:8000` | Hybrid retrieval service |
| `REASONING_URL` | `http://localhost:8001` | `http://reasoning:8000` | LangGraph reasoning service |
| `VALIDATOR_URL` | `http://localhost:8004` | `http://validator:8000` | Answer validator service |
| `HOST` | `0.0.0.0` | `0.0.0.0` | Bind host |
| `PORT` | `8003` | `8000` | Orchestrator port |

---

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   pytest tests/ -v
   ```

3. Start server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8003 --reload
   ```
