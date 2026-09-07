from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langfuse import Langfuse

from .schemas import (
    BenchmarkItem,
    PerItemResult,
    RunBenchmarkRequest,
    RunBenchmarkResponse,
)
from .scoring import score_prediction, compute_page_retrieval_metrics

app = FastAPI(title="eval-service", version="0.2.0")

BENCHMARK_PATH = os.getenv(
    "BENCHMARK_PATH",
    "services/eval-service/data/benchmark_100.json",
)

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
ORCHESTRATOR_URL_DEFAULT = os.getenv("ORCHESTRATOR_URL", "http://localhost:8003")
RETRIEVAL_URL_DEFAULT = os.getenv("RETRIEVAL_URL")  # optional, for retrieval metrics later


def get_langfuse() -> Langfuse | None:
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        return Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    return None


def load_benchmark() -> list[BenchmarkItem]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [BenchmarkItem.model_validate(x) for x in raw]


def call_orchestrator(orchestrator_url: str, question_text: str, question_id: str, document_id: str | None = None, debug: bool = False) -> dict[str, Any]:
    """
    Calls Orchestrator POST /run.

    Orchestrator request:
      {"question": "...", "session_id": "...", "document_id": optional}

    Returns orchestrator JSON response:
      {answer_type, evidence, params, validation_status, latency_ms, ...}
    """
    url = f"{orchestrator_url.rstrip('/')}/run"
    payload = {"question": question_text, "session_id": question_id}
    if document_id:
        payload["document_id"] = document_id
    if debug:
        payload["debug"] = True

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()



def _coerce_page_to_int(p) -> int | None:
    if p is None:
        return None
    if isinstance(p, int):
        return p
    if isinstance(p, list) and p and isinstance(p[0], int):
        return p[0]
    # sometimes string
    try:
        return int(p)
    except Exception:
        return None


def extract_retrieved_pages(orchestrator_response: dict[str, Any]) -> list[tuple[str, int]]:
    """
    Option-A ready: tries to extract ranked retrieved candidates from orchestrator response.

    Supported shapes (any of these):
    - response["retrieval_debug"]["calls"][i]["candidates"][j] with keys {document_id, page}
    - response["retrieval_debug"]["candidates"][j]
    - response["retrieved_candidates"][j]
    """
    retrieved_pages: list[tuple[str, int]] = []

    rd = orchestrator_response.get("retrieval_debug") or {}
    calls = rd.get("calls")

    def add_candidate(c: dict[str, Any]):
        doc_id = c.get("document_id") or c.get("doc_uid")
        page_i = _coerce_page_to_int(c.get("page"))
        if doc_id and page_i is not None:
            retrieved_pages.append((str(doc_id), int(page_i)))

    if isinstance(calls, list):
        for call in calls:
            cands = call.get("candidates", [])
            if isinstance(cands, list):
                for c in cands:
                    if isinstance(c, dict):
                        add_candidate(c)

    # fallback: rd.candidates
    cands = rd.get("candidates")
    if isinstance(cands, list):
        for c in cands:
            if isinstance(c, dict):
                add_candidate(c)

    # fallback: top-level retrieved_candidates
    top = orchestrator_response.get("retrieved_candidates")
    if isinstance(top, list):
        for c in top:
            if isinstance(c, dict):
                add_candidate(c)

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in retrieved_pages:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/langfuse_status")
def langfuse_status():
    return {
        "langfuse_enabled": bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY),
        "langfuse_host": LANGFUSE_HOST or "default",
    }


@app.post("/langfuse_ping")
def langfuse_ping():
    lf = get_langfuse()
    if not lf:
        raise HTTPException(status_code=500, detail="Langfuse not configured")

    try:
        t = lf.trace(name="ping", input={"ok": True})

        # Create an observation so it appears in the Tracing UI
        span = t.span(name="ping_span", input={"step": "start"})
        if hasattr(span, "end"):
            span.end(output={"step": "done"})
        else:
            span.update(output={"step": "done"})

        t.score(name="ping_score", value=1.0)
        t.update(output={"status": "sent"})
        lf.flush()

        return {"sent": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"sent": False, "error": repr(e)})



@app.get("/benchmark_info")
def benchmark_info():
    try:
        items = load_benchmark()
        return {"benchmark_path": BENCHMARK_PATH, "num_items": len(items)}
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": "benchmark file not found", "benchmark_path": BENCHMARK_PATH},
        )


@app.post("/run_benchmark", response_model=RunBenchmarkResponse)
def run_benchmark(req: RunBenchmarkRequest) -> RunBenchmarkResponse:
    items = load_benchmark()
    if req.limit:
        items = items[: req.limit]

    lf = get_langfuse()
    results: list[PerItemResult] = []

    ems: list[float] = []
    f1s: list[float] = []
    numeric_flags: list[bool] = []

    # Retrieval metrics accumulators (page-level)
    hits_k: list[float] = []
    recalls_k: list[float] = []
    precisions_k: list[float] = []
    rrs: list[float] = []

    for item in items:
        t0 = time.time()

        per = PerItemResult(
            question_id=item.question_id,
            question_text=item.question_text,
            is_answerable=item.is_answerable,
            ground_truth_answer=item.ground_truth_answer,
        )

        # Gold evidence pages (doc_uid + page)
        gold_pages = [
            (e.source_doc_uid, e.source_page)
            for e in item.gold_evidence
            if e.source_doc_uid and e.source_page is not None
        ]
        # TODO: once orchestrator/retrieval integration is done, fill retrieved_pages from actual retrieval output
        retrieved_pages: list[tuple[str, int]] = []

        trace = None
        if lf:
            trace = lf.trace(
                name="benchmark_item",
                input={"question_id": item.question_id, "question": item.question_text},
                metadata={"langfuse_project": req.langfuse_project},
            )

        try:
            orch_url = req.orchestrator_url or ORCHESTRATOR_URL_DEFAULT
            pred = call_orchestrator(
                orchestrator_url=orch_url,
                question_text=item.question_text,
                question_id=item.question_id,
                debug=req.debug,
            )

            per.predicted_answer = pred
            per.predicted_answer_type = str(pred.get("answer_type"))

            # If orchestrator returned retrieval candidates (Option A), compute retrieval metrics
            if req.debug:
                retrieved_pages = extract_retrieved_pages(pred)

            # Extract predicted value based on strict answer schema
            at = str(pred.get("answer_type", "insufficient_evidence"))
            params = pred.get("params") or {}

            if at in ("direct", "calculated"):
                predicted_value = params.get("value")
            elif at == "multi_span":
                predicted_value = params.get("values")
            else:
                predicted_value = None  # insufficient_evidence or unknown

            # Special case: unanswerable gold + insufficient prediction => correct
            if (item.is_answerable is False) and (at == "insufficient_evidence"):
                em, f1, num_ok = 1.0, 1.0, None
            else:
                em, f1, num_ok = score_prediction(predicted_value, item.ground_truth_answer, item.scale)

            per.em, per.f1, per.numeric_ok = em, f1, num_ok
            ems.append(em)
            f1s.append(f1)
            if num_ok is not None:
                numeric_flags.append(num_ok)

            # Page-level retrieval metrics (only if retrieved_pages is populated)
            if retrieved_pages and gold_pages:
                m = compute_page_retrieval_metrics(retrieved_pages, gold_pages, k=req.retrieval_k)
                per.retrieval_hit = m['hit']
                per.retrieval_recall = m['recall']
                per.retrieval_precision = m['precision']
                per.retrieval_rr = m['rr']
                hits_k.append(m['hit'])
                recalls_k.append(m['recall'])
                precisions_k.append(m['precision'])
                rrs.append(m['rr'])

            if trace:
                trace.score(name="em", value=em)
                trace.score(name="f1", value=f1)
                if num_ok is not None:
                    trace.score(name="numeric_ok", value=1.0 if num_ok else 0.0)

        except Exception as e:
            per.error = str(e)
            if trace:
                trace.score(name="error", value=1.0)
        finally:
            if trace:
                trace.update(output={"result": per.model_dump()})
            _ = t0  # reserved for latency later

        results.append(per)

    avg_em = sum(ems) / len(ems) if ems else None
    avg_f1 = sum(f1s) / len(f1s) if f1s else None
    numeric_acc = sum(1 for x in numeric_flags if x) / len(numeric_flags) if numeric_flags else None

    avg_hit_at_k = (sum(hits_k) / len(hits_k)) if hits_k else None
    avg_recall_at_k = (sum(recalls_k) / len(recalls_k)) if recalls_k else None
    avg_precision_at_k = (sum(precisions_k) / len(precisions_k)) if precisions_k else None
    mrr_at_k = (sum(rrs) / len(rrs)) if rrs else None

    if lf:
        lf.flush()

    return RunBenchmarkResponse(
        num_items=len(items),
        num_scored=len(ems),
        avg_em=avg_em,
        avg_f1=avg_f1,
        numeric_accuracy=numeric_acc,
        avg_hit_at_k=avg_hit_at_k,
        avg_recall_at_k=avg_recall_at_k,
        avg_precision_at_k=avg_precision_at_k,
        mrr_at_k=mrr_at_k,
        results=results,
    )
