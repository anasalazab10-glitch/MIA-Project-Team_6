"""
Pydantic schemas for the LEDGER Orchestrator Service.
Defines strict request/response formats conforming to the project specification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Question Answering / Run Schemas
# ---------------------------------------------------------

class RunRequest(BaseModel):
    question: str = Field(..., description="Natural language question to ask the system")
    session_id: Optional[str] = Field(None, description="Optional session or conversation ID")
    document_id: Optional[str] = Field(
        None, description="Optional document ID for document/company scoping"
    )


class EvidenceCitation(BaseModel):
    document_id: str
    page: Union[int, List[int], str]
    section: Optional[str] = None


class StrictAnswer(BaseModel):
    answer_type: Literal["direct", "calculated", "multi_span", "insufficient_evidence"]
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    answer_type: str
    evidence: List[Dict[str, Any]]
    params: Dict[str, Any]
    validation_status: str = "valid"
    validation_reason: Optional[str] = None
    latency_ms: float = 0.0


# ---------------------------------------------------------
# Document Ingestion & Catalog Schemas
# ---------------------------------------------------------

class IngestResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    num_pages: int
    num_elements: int
    num_tables: int
    message: str


class TableSummary(BaseModel):
    document_id: str
    page: Union[int, List[int], str]
    section: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    num_rows: int = 0
    preview: List[List[str]] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_elements: int
    num_tables: int
    created_at: str


class DocumentDetail(DocumentSummary):
    tables: List[TableSummary] = Field(default_factory=list)
    elements_preview: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------
# Dashboard & Observability Schemas
# ---------------------------------------------------------

class QueryAuditItem(BaseModel):
    query_id: str
    question: str
    answer_type: str
    validation_passed: bool
    latency_ms: float
    timestamp: str
    evidence_count: int = 0


class DashboardStats(BaseModel):
    total_documents: int
    total_tables: int
    total_elements: int
    documents: List[DocumentSummary]
    tables: List[TableSummary]
    recent_queries: List[QueryAuditItem]


# ---------------------------------------------------------
# Health & Status
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str = "orchestrator-api"
    downstream_services: Dict[str, Dict[str, Any]]
