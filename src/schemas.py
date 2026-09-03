from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    """Supported content types for ingested elements/chunks.
    
    Ensures clear separation between standard prose, tabular financial data,
    and structural headings.
    """
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


class RetrievalMethod(str, Enum):
    """Retrieval strategies used across the search pipeline.
    
    Allows tagging individual candidates and responses so downstream components
    and Langfuse tracing know which retriever or fusion stage produced them.
    """
    DENSE = "dense"        # Vector similarity search (e.g., Qdrant / embeddings)
    BM25 = "bm25"          # Lexical/keyword search
    RRF = "rrf"            # Reciprocal Rank Fusion of dense + BM25
    RERANKER = "reranker"  # Cross-encoder reranked output
    HYBRID = "hybrid"      # End-to-end hybrid pipeline (BM25 + Dense -> RRF -> Rerank)


# ---------------------------------------------------------------------------
# Citation Schema (Aligned with Project Strict Answer Schema)
# ---------------------------------------------------------------------------

class EvidenceCitation(BaseModel):
    """Strict citation format required by the project's answer-validator-api.
    
    The reasoning agent must include evidence in this exact format:
    {"document_id": "...", "page": int, "section": "..."}
    """
    document_id: str
    page: int
    section: str | None = None


# ---------------------------------------------------------------------------
# Chunk Content Models
# ---------------------------------------------------------------------------

class TableContent(BaseModel):
    """Structured representation for financial tables.
    
    Provides structured access for table-aware reasoning and a helper
    to serialize into Markdown text for BM25 indexing and Cross-Encoder reranking.
    """
    headers: list[str]
    rows: list[list[str]]

    def to_text(self) -> str:
        """Convert table into clean Markdown format for text-based retrieval."""
        header_line = "| " + " | ".join(self.headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        row_lines = ["| " + " | ".join(row) + " |" for row in self.rows]
        return "\n".join([header_line, sep_line] + row_lines)


class Chunk(BaseModel):
    """A retrieval-ready document chunk.
    
    Preserves all required metadata (document_id, page, section, content_type)
    as specified in the project PDF, with helpers for text extraction and citation.
    """
    chunk_id: str
    document_id: str
    page: int
    section: str | None = None
    content_type: ContentType
    content: str | TableContent
    bbox: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Return a string representation of chunk content.
        
        Crucial for BM25 tokenizers and Cross-Encoders, which require string
        inputs regardless of whether the chunk is prose or a structured table.
        """
        if isinstance(self.content, TableContent):
            return self.content.to_text()
        return str(self.content)

    def to_evidence(self) -> EvidenceCitation:
        """Helper to create an exact EvidenceCitation for the answer validator."""
        return EvidenceCitation(
            document_id=self.document_id,
            page=self.page,
            section=self.section,
        )


# ---------------------------------------------------------------------------
# Retrieval Candidates & Fusion
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    """A single retrieved chunk candidate.
    
    Tracks primary score and rank, plus historical scores (BM25, dense, RRF,
    reranker) and initial ranks to support required observability and evaluation
    (evaluating reranker value and stage latency in Langfuse).
    """
    chunk: Chunk
    score: float
    retrieval_method: RetrievalMethod
    rank: int
    initial_rank: int | None = None
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Tracks intermediate scores across stages (e.g., {'bm25': 12.3, 'dense': 0.81, 'rrf': 0.03, 'reranker': 2.45})"
    )


# ---------------------------------------------------------------------------
# API Request & Response Schemas
# ---------------------------------------------------------------------------

class RetrievalRequest(BaseModel):
    """Standard request payload for retrieval endpoints."""
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.HYBRID)
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters (e.g., {'document_id': 'doc1', 'year': 2024})"
    )


class RerankRequest(BaseModel):
    """Request payload to rerank an existing candidate list (e.g. top 30 -> top 5)."""
    query: str
    candidates: list[Candidate]
    top_k: int = Field(default=5, ge=1, le=50)


class RetrievalResponse(BaseModel):
    """Standard response containing query, method used, and ranked candidates."""
    query: str
    retrieval_method: RetrievalMethod
    candidates: list[Candidate]