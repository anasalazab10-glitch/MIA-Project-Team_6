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


def call_orchestrator(orchestrator_url: str, question_text: str) -> dict[str, Any]:
    """
    TODO: update once orchestrator endpoints are finalized.
    Expected orchestrator behavior: accept a natural language question and return strict answer JSON.
    """
    raise NotImplementedError(
        "Orchestrator endpoint not integrated yet. "
        "Provide orchestrator contract and implement call_orchestrator()."
    )


@app.get("/health")
def health():
    return {"status": "ok"}


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
            if not req.orchestrator_url:
                raise HTTPException(status_code=501, detail="orchestrator_url not provided yet")

            # TODO: enable when orchestrator contract is known
            pred = call_orchestrator(req.orchestrator_url, item.question_text)

            per.predicted_answer = pred
            # scoring expects predicted_value; once strict schema is known, extract from pred
            predicted_value = pred.get("params", {}).get("value", pred)
            em, f1, num_ok = score_prediction(predicted_value, item.ground_truth_answer, item.scale)

            per.em, per.f1, per.numeric_ok = em, f1, num_ok
            ems.append(em)
            f1s.append(f1)
            if num_ok is not None:
                numeric_flags.append(num_ok)

            # Page-level retrieval metrics (only if retrieved_pages is populated)
            if retrieved_pages and gold_pages:
                m = compute_page_retrieval_metrics(retrieved_pages, gold_pages, k=5)
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
                trace.end(output={"result": per.model_dump()})
            _ = t0  # reserved for latency later

        results.append(per)

    avg_em = sum(ems) / len(ems) if ems else None
    avg_f1 = sum(f1s) / len(f1s) if f1s else None
    numeric_acc = sum(1 for x in numeric_flags if x) / len(numeric_flags) if numeric_flags else None

    avg_hit_at_k = (sum(hits_k) / len(hits_k)) if hits_k else None
    avg_recall_at_k = (sum(recalls_k) / len(recalls_k)) if recalls_k else None
    avg_precision_at_k = (sum(precisions_k) / len(precisions_k)) if precisions_k else None
    mrr_at_k = (sum(rrs) / len(rrs)) if rrs else None

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
