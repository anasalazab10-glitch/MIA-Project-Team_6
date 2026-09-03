# Schemas Design & Technical Rationale (`src/schemas.py`)

This document outlines the architecture of the data contracts defined in [`src/schemas.py`] and explains why each model, modification, and addition was necessary for both independent development and end-to-end integration.

---

## 1. Context & Team Responsibilities

- **Mariam's (Dense Retrieval)**:
  - Document chunking, embeddings, Qdrant vector indexing, and dense retrieval returning top 30 candidates.
- **Justin's (Hybrid Retrieval & Reranking)**:
  - BM25 keyword retrieval.
  - Candidate merging / Reciprocal Rank Fusion (RRF).
  - Cross-Encoder reranking (Top 30 $\to$ Final Top 5).
  - Preparing citation evidence for the downstream reasoning agent and answer validator.

Both sub-teams operate against the common `Chunk` contract and `mock_chunks.json` before live PDFs are connected.

---

## 2. Breakdown of Changes & Additions

### A. Table Serialization (`TableContent.to_text()` & `Chunk.to_text()`)

* **What was added:**
  Added the `to_text()` method to both `TableContent` and `Chunk`.
  ```python
  class TableContent(BaseModel):
      headers: list[str]
      rows: list[list[str]]

      def to_text(self) -> str:
          # Formats into standard Markdown table string
  ```
* **Why it is necessary:**
  - In financial documents (TAT-DQA), critical facts reside in tables (balance sheets, income statements, regional breakdowns).
  - BM25 tokenizers and Cross-Encoder reranker models (`sentence-transformers/CrossEncoder`) accept **pure text/strings** for scoring `(query, document)`.
  - Without a deterministic `to_text()` method, passing a `TableContent` object into BM25 or a Cross-Encoder causes type errors or requires ad-hoc formatting logic scattered throughout the codebase.
  - Markdown table formatting preserves row and column relationships for both lexical matching and cross-attention semantic scoring.

---

### B. Candidate Traceability & Observability (`Candidate.initial_rank` & `Candidate.scores`)

* **What was added:**
  Added `initial_rank` and a `scores` dictionary to `Candidate`:
  ```python
  class Candidate(BaseModel):
      chunk: Chunk
      score: float
      retrieval_method: RetrievalMethod
      rank: int
      initial_rank: int | None = None
      scores: dict[str, float] = Field(default_factory=dict)
  ```
* **Why it is necessary:**
  - **Project Specification Requirement**: Page 3 mandates:
    > *"Candidates should be over-retrieved and reranked before being handed to the agent (e.g. top 30 → reranker → top 5); the value of reranking should be evaluated, not assumed."*
  - **Langfuse Tracing Requirement**: Page 3 & 6 require tracing each step (BM25 vs. Dense vs. RRF vs. Rerank).
  - When fusing 30 Dense candidates and 30 BM25 candidates with RRF, a candidate has an initial dense rank, a BM25 rank, an RRF score, and later a reranker score.
  - Storing these in `scores` (e.g., `{"bm25": 14.5, "dense": 0.82, "rrf": 0.031, "reranker": 2.89}`) enables measuring whether reranking actually promoted the ground-truth chunk into the Top 5 without losing historical context.

---

### C. Citation Contract Alignment (`EvidenceCitation` & `Chunk.to_evidence()`)

* **What was added:**
  Defined `EvidenceCitation` and added `Chunk.to_evidence()`:
  ```python
  class EvidenceCitation(BaseModel):
      document_id: str
      page: int
      section: str | None = None

  class Chunk(BaseModel):
      ...
      def to_evidence(self) -> EvidenceCitation:
          return EvidenceCitation(
              document_id=self.document_id,
              page=self.page,
              section=self.section,
          )
  ```
* **Why it is necessary:**
  - **Answer Validator Rule**: Pages 4–6 of the project specification state that every answer produced by the agent must include evidence strictly adhering to:
    ```json
    { "document_id": "...", "page": 0, "section": "..." }
    ```
  - Providing this helper ensures retrieval candidates directly map into the exact citation format needed by downstream services (`agent-service` and `answer-validator-api`).

---

### D. API Request Contracts (`RetrievalRequest` & `RerankRequest`)

* **What was added:**
  ```python
  class RetrievalRequest(BaseModel):
      query: str
      top_k: int = Field(default=5, ge=1, le=100)
      retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.HYBRID)
      metadata_filter: dict[str, Any] | None = None

  class RerankRequest(BaseModel):
      query: str
      candidates: list[Candidate]
      top_k: int = Field(default=5, ge=1, le=50)
  ```
* **Why it is necessary:**
  - The project specifies a microservice architecture (`retrieval-api` running on FastAPI).
  - Having explicit Pydantic request models enables:
    1. Independent endpoint testing (e.g. `/retrieve/bm25`, `/retrieve/dense`, `/rerank`).
    2. Decoupled testing: our reranker can be tested by receiving a list of candidates independently of how they were generated.
    3. Input validation, default values, and automatic OpenAPI documentation.

---

## 3. Compatibility Summary

| Model / Field | Status | Impact on Project |
| :--- | :--- | :--- |
| `ContentType` | Existing | None (Fully preserved) |
| `RetrievalMethod` | Existing | Added explanatory comments, preserved enum keys |
| `TableContent` | Modified | Added `.to_text()` helper (non-breaking) |
| `Chunk` | Modified | Added `.to_text()` and `.to_evidence()` helpers (non-breaking) |
| `Candidate` | Modified | Added optional `initial_rank` and `scores` with defaults (non-breaking) |
| `EvidenceCitation` | **New** | Clean contract for answer validation |
| `RetrievalRequest` | **New** | Required for FastAPI endpoint inputs |
| `RerankRequest` | **New** | Allows calling reranker as an isolated step |
| `RetrievalResponse`| Existing | None (Fully preserved) |
