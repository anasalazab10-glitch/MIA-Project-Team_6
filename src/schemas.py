from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Add other content types here if the document processor needs them.
class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


# Retrieval methods used in the retrieval pipeline.
class RetrievalMethod(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    RRF = "rrf"
    RERANKER = "reranker"
    HYBRID = "hybrid"


class TableContent(BaseModel):
    headers: list[str]
    rows: list[list[str]]


# A retrieval-ready document chunk.
class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    page: int
    section: str | None = None
    content_type: ContentType
    content: str | TableContent
    bbox: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# A chunk returned by a retrieval method.
class Candidate(BaseModel):
    chunk: Chunk
    score: float
    retrieval_method: RetrievalMethod
    rank: int


# The response containing the retrieved candidates.
class RetrievalResponse(BaseModel):
    query: str
    retrieval_method: RetrievalMethod
    candidates: list[Candidate]