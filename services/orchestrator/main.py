"""
LEDGER Central Orchestrator Service (orchestrator-api)
Routes documents and questions between Document Processor, Retrieval,
Reasoning Brain, Answer Validator, and Gradio UI / Eval services.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from clients import ServiceClients
from config import settings
from metadata_store import MetadataStore
from schemas import (
    DashboardStats,
    DocumentDetail,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    RunRequest,
    RunResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")

clients = ServiceClients()
metadata_store = MetadataStore(data_dir=settings.data_dir)

def _coerce_page_to_int(p: Any) -> Any:
    if p is None:
        return None
    if isinstance(p, int):
        return p
    if isinstance(p, list) and p:
        return _coerce_page_to_int(p[0])
    if isinstance(p, str):
        try:
            return int(p)
        except Exception:
            return p
    return p


def _normalize_evidence(evidence: Any) -> list[dict]:
    if not isinstance(evidence, list):
        return []
    out: list[dict] = []
    for c in evidence:
        if not isinstance(c, dict):
            continue
        c2 = dict(c)
        c2["page"] = _coerce_page_to_int(c2.get("page"))
        out.append(c2)
    return out

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing LEDGER Orchestrator Service...")
    await clients.start()
    logger.info(f"Target DocProcessor: {settings.doc_processor_url}")
    logger.info(f"Target Retrieval:    {settings.retrieval_url}")
    logger.info(f"Target Reasoning:    {settings.reasoning_url}")
    logger.info(f"Target Validator:    {settings.validator_url}")
    yield
    logger.info("Shutting down LEDGER Orchestrator Service...")
    await clients.close()


app = FastAPI(
    title="LEDGER Orchestrator API",
    description="Central coordination engine for Project LEDGER financial document intelligence agent.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 1. Health & Status
# ----------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    downstream = await clients.check_health()
    all_ok = all(s.get("reachable", False) for s in downstream.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        service="orchestrator-api",
        downstream_services=downstream,
    )


# ----------------------------------------------------------------------
# 2. Question Answering / Query Execution
# ----------------------------------------------------------------------

@app.post("/run", response_model=RunResponse)
async def run_query(request: RunRequest):
    """
    Main QA endpoint:
    1. Forwards question to Reasoning Agent (LangGraph)
    2. Sends generated answer to Validator Service
    3. Rejects hallucinated/unverified answers with insufficient_evidence
    4. Logs audit metrics & latency
    5. Returns strict JSON answer
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()
    logger.info(f"[Query Received] Question: '{question}' (session_id={request.session_id})")

    # Step 1: Execute reasoning agent
    try:
        raw_answer = await clients.run_reasoning(
            question=question,
            session_id=request.session_id,
            document_id=request.document_id,
        )
    except Exception as exc:
        logger.error(f"[Reasoning Error] {exc}")
        # Graceful fallback: return insufficient_evidence rather than a 500 error
        raw_answer = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {
                "reason": f"Reasoning service was unable to answer: {str(exc)}"
            },
        }

    # Normalize answer payload
    answer_type = raw_answer.get("answer_type", "insufficient_evidence")
    evidence = _normalize_evidence(raw_answer.get("evidence", []))
    params = raw_answer.get("params", {})
    if not isinstance(params, dict):
        params = {}
    retrieved_candidates = raw_answer.get("retrieved_candidates", [])


    validation_payload = {
        "answer_type": answer_type,
        "evidence": evidence,
        "params": params,
    }

    # Step 2: Validate against Strict Answer Schema via Answer-Validator-API
    val_result = await clients.validate_answer(validation_payload)
    is_valid = val_result.get("valid", False)
    val_reason = val_result.get("reason", "")

    if is_valid:
        logger.info(
            f"[ANSWER-VALIDATOR-SUCCESS] Validated answer of type '{answer_type}' "
            f"with {len(evidence)} evidence citations and {len(retrieved_candidates)} retrieved candidates."
        )
        final_answer = validation_payload
        validation_status = "valid"
    else:
        logger.warning(
            f"[ANSWER-VALIDATOR-ERROR] Invalid answer for '{answer_type}'. Reason: {val_reason}"
        )
        # CRITICAL SAFETY RULE: Never return an unverified or invalid answer to the user.
        # Strict fallback to insufficient_evidence explaining the validation failure.
        final_answer = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {
                "reason": f"Answer rejected by validator: {val_reason}"
            },
        }
        validation_status = "rejected"

    elapsed_ms = (time.time() - t0) * 1000.0

    # Step 3: Record query in local audit store for the dashboard
    metadata_store.record_query(
        question=question,
        answer_type=final_answer.get("answer_type", "unknown"),
        validation_passed=is_valid,
        latency_ms=elapsed_ms,
        evidence_count=len(final_answer.get("evidence", [])),
        candidates_count=len(retrieved_candidates),
    )

    return RunResponse(
        answer_type=final_answer["answer_type"],
        evidence=final_answer.get("evidence", []),
        params=final_answer.get("params", {}),
        retrieved_candidates=retrieved_candidates,
        validation_status=validation_status,
        validation_reason=val_reason if not is_valid else None,
        latency_ms=round(elapsed_ms, 2),
    )


# ----------------------------------------------------------------------
# 3. Document Ingestion Pipeline
# ----------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Raw PDF financial document"),
    document_id: Optional[str] = Form(None, description="Optional custom document ID"),
    dpi: int = Form(200, description="DPI for PDF rendering in doc processor"),
):
    """
    Ingestion endpoint:
    1. Reads raw PDF upload
    2. Forwards to doc-processor-api (/process) for layout, text, tables, and bounding boxes
    3. Indexes structured elements into retrieval-api (/index)
    4. Records metadata, detected tables, and elements in local catalog
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(f"[Ingest] Processing '{file.filename}' ({len(pdf_bytes)} bytes)...")

    # Call doc processor
    try:
        proc_result = await clients.process_document(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            document_id=document_id,
            dpi=dpi,
        )
    except Exception as exc:
        logger.error(f"[Ingest Error] DocProcessor failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Document processor failed: {exc}")

    doc_id = proc_result.get("document_id", file.filename)
    num_pages = proc_result.get("num_pages", 1)
    elements = proc_result.get("elements", [])

    # Index into retrieval service
    index_res = await clients.index_elements(document_id=doc_id, elements=elements)
    logger.info(f"[Ingest] Retrieval indexing result for '{doc_id}': {index_res}")

    # Register in metadata store
    doc_detail = metadata_store.record_document(
        document_id=doc_id,
        filename=file.filename,
        num_pages=num_pages,
        elements=elements,
    )

    logger.info(
        f"[Ingest Complete] '{doc_id}' indexed: {num_pages} pages, "
        f"{doc_detail.num_elements} elements, {doc_detail.num_tables} tables."
    )

    return IngestResponse(
        status="indexed",
        document_id=doc_id,
        filename=file.filename,
        num_pages=num_pages,
        num_elements=doc_detail.num_elements,
        num_tables=doc_detail.num_tables,
        message=f"Successfully processed and indexed document with {doc_detail.num_tables} detected tables.",
    )


# ----------------------------------------------------------------------
# 4. Document & Catalog Endpoints
# ----------------------------------------------------------------------

@app.get("/documents", response_model=List[DocumentSummary])
def list_documents():
    """Returns a list of all indexed documents."""
    return metadata_store.list_documents()


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str):
    """Returns details and extracted tables for a specific document."""
    doc = metadata_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return doc


# ----------------------------------------------------------------------
# 5. Dashboard Corpus Statistics (for Gradio UI)
# ----------------------------------------------------------------------

@app.get("/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats():
    """
    Returns corpus-level statistics for the Gradio UI dashboard:
    - Total indexed documents
    - Document list
    - Detected financial tables
    - Recent queries with latency and validation status
    """
    return metadata_store.get_dashboard_stats()
