
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
from .scoring import (
    score_prediction,
    compute_page_retrieval_metrics,
)


app = FastAPI(
    title="eval-service",
    version="0.2.0",
)


# ============================================================================
# Configuration
# ============================================================================

BENCHMARK_PATH = os.getenv(
    "BENCHMARK_PATH",
    "services/eval-service/data/benchmark_100.json",
)

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

ORCHESTRATOR_URL_DEFAULT = os.getenv(
    "ORCHESTRATOR_URL",
    "http://localhost:8003",
)

RETRIEVAL_URL_DEFAULT = os.getenv(
    "RETRIEVAL_URL"
)

# The directory will be created automatically.
RESULTS_DIR = os.getenv(
    "RESULTS_DIR",
    "/app/results",
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "benchmark_results.json",
)


# ============================================================================
# Langfuse
# ============================================================================

def get_langfuse() -> Langfuse | None:
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        return Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )

    return None


# ============================================================================
# Benchmark loading
# ============================================================================

def load_benchmark() -> list[BenchmarkItem]:
    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    return [
        BenchmarkItem.model_validate(item)
        for item in raw
    ]


# ============================================================================
# Orchestrator
# ============================================================================

def call_orchestrator(
    orchestrator_url: str,
    question_text: str,
    question_id: str,
    document_id: str | None = None,
    debug: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """
    Calls the Orchestrator POST /run endpoint.
    """

    url = f"{orchestrator_url.rstrip('/')}/run"

    payload: dict[str, Any] = {
        "question": question_text,
        "session_id": question_id,
    }

    if document_id:
        payload["document_id"] = document_id

    if debug:
        payload["debug"] = True

    if trace_id:
        payload["trace_id"] = trace_id

    response = requests.post(
        url,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================================
# Retrieval result extraction
# ============================================================================

def _coerce_page_to_int(
    page: Any,
) -> int | None:

    if page is None:
        return None

    if isinstance(page, int):
        return page

    if isinstance(page, list) and page:
        if isinstance(page[0], int):
            return page[0]

    try:
        return int(page)
    except Exception:
        return None


def extract_retrieved_pages(
    orchestrator_response: dict[str, Any],
) -> list[tuple[str, int]]:
    """
    Extract document/page pairs from retrieved candidates.

    Supported response shapes:

    1. response["retrieval_debug"]["calls"][i]["candidates"]

    2. response["retrieval_debug"]["candidates"]

    3. response["retrieved_candidates"]
    """

    retrieved_pages: list[tuple[str, int]] = []

    retrieval_debug = (
        orchestrator_response.get("retrieval_debug")
        or {}
    )

    calls = retrieval_debug.get("calls")

    def add_candidate(
        candidate: dict[str, Any],
    ) -> None:

        document_id = (
            candidate.get("document_id")
            or candidate.get("doc_uid")
        )

        page = _coerce_page_to_int(
            candidate.get("page")
        )

        if document_id and page is not None:
            retrieved_pages.append(
                (
                    str(document_id),
                    int(page),
                )
            )

    # ------------------------------------------------------------------------
    # Option 1:
    # retrieval_debug.calls[*].candidates
    # ------------------------------------------------------------------------

    if isinstance(calls, list):

        for call in calls:

            candidates = call.get(
                "candidates",
                [],
            )

            if isinstance(candidates, list):

                for candidate in candidates:

                    if isinstance(candidate, dict):
                        add_candidate(candidate)

    # ------------------------------------------------------------------------
    # Option 2:
    # retrieval_debug.candidates
    # ------------------------------------------------------------------------

    candidates = retrieval_debug.get(
        "candidates"
    )

    if isinstance(candidates, list):

        for candidate in candidates:

            if isinstance(candidate, dict):
                add_candidate(candidate)

    # ------------------------------------------------------------------------
    # Option 3:
    # top-level retrieved_candidates
    # ------------------------------------------------------------------------

    top_candidates = orchestrator_response.get(
        "retrieved_candidates"
    )

    if isinstance(top_candidates, list):

        for candidate in top_candidates:

            if isinstance(candidate, dict):
                add_candidate(candidate)

    # ------------------------------------------------------------------------
    # Remove duplicates while preserving order
    # ------------------------------------------------------------------------

    seen: set[tuple[str, int]] = set()

    unique_pages: list[tuple[str, int]] = []

    for page in retrieved_pages:

        if page not in seen:

            seen.add(page)
            unique_pages.append(page)

    return unique_pages


# ============================================================================
# Health
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================================
# Langfuse status
# ============================================================================

@app.get("/langfuse_status")
def langfuse_status():

    return {
        "langfuse_enabled": bool(
            LANGFUSE_PUBLIC_KEY
            and LANGFUSE_SECRET_KEY
        ),
        "langfuse_host": (
            LANGFUSE_HOST
            or "default"
        ),
    }


# ============================================================================
# Langfuse ping
# ============================================================================

@app.post("/langfuse_ping")
def langfuse_ping():

    lf = get_langfuse()

    if not lf:

        raise HTTPException(
            status_code=500,
            detail="Langfuse not configured",
        )

    try:

        trace = lf.trace(
            name="ping",
            input={"ok": True},
        )

        span = trace.span(
            name="ping_span",
            input={"step": "start"},
        )

        if hasattr(span, "end"):

            span.end(
                output={"step": "done"}
            )

        else:

            span.update(
                output={"step": "done"}
            )

        trace.score(
            name="ping_score",
            value=1.0,
        )

        trace.update(
            output={"status": "sent"}
        )

        lf.flush()

        return {
            "sent": True
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "sent": False,
                "error": repr(e),
            },
        )


# ============================================================================
# Benchmark information
# ============================================================================

@app.get("/benchmark_info")
def benchmark_info():

    try:

        items = load_benchmark()

        return {
            "benchmark_path": BENCHMARK_PATH,
            "num_items": len(items),
        }

    except FileNotFoundError:

        return JSONResponse(
            status_code=404,
            content={
                "error": "benchmark file not found",
                "benchmark_path": BENCHMARK_PATH,
            },
        )


# ============================================================================
# Run benchmark
# ============================================================================

@app.post(
    "/run_benchmark",
    response_model=RunBenchmarkResponse,
)
def run_benchmark(
    req: RunBenchmarkRequest,
) -> RunBenchmarkResponse:

    # ========================================================================
    # Load benchmark
    # ========================================================================

    all_items = load_benchmark()

    total_available = len(all_items)

    # ========================================================================
    # Select questions
    #
    # Priority:
    #
    # 1. start/end
    # 2. limit
    # 3. all questions
    # ========================================================================

    if req.start is not None or req.end is not None:

        start = (
            req.start
            if req.start is not None
            else 1
        )

        end = (
            req.end
            if req.end is not None
            else total_available
        )

        # Keep start/end within valid bounds.
        start = max(1, start)
        end = min(total_available, end)

        if start > end:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid range: start={start}, "
                    f"end={end}"
                ),
            )

        items = all_items[
            start - 1:end
        ]

    elif req.limit:

        if req.limit < 1:

            raise HTTPException(
                status_code=400,
                detail="limit must be greater than 0",
            )

        items = all_items[
            :req.limit
        ]

    else:

        items = all_items

    # ========================================================================
    # Handle empty benchmark
    # ========================================================================

    if not items:

        raise HTTPException(
            status_code=400,
            detail="No benchmark questions selected",
        )

    # ========================================================================
    # Progress information
    # ========================================================================

    total_questions = len(items)

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "LEDGER BENCHMARK STARTED",
        flush=True,
    )

    print(
        f"Selected questions: {total_questions}",
        flush=True,
    )

    print(
        f"Total available: {total_available}",
        flush=True,
    )

    print(
        f"Results file: {RESULTS_FILE}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    # ========================================================================
    # Initialize Langfuse
    # ========================================================================

    lf = get_langfuse()

    # ========================================================================
    # Result containers
    # ========================================================================

    results: list[PerItemResult] = []

    ems: list[float] = []
    f1s: list[float] = []
    numeric_flags: list[bool] = []

    # Retrieval metrics
    hits_k: list[float] = []
    recalls_k: list[float] = []
    precisions_k: list[float] = []
    rrs: list[float] = []

    benchmark_start_time = time.time()

    # ========================================================================
    # Process questions
    # ========================================================================

    for index, item in enumerate(
        items,
        start=1,
    ):

        question_start_time = time.time()

        # --------------------------------------------------------------------
        # Question start message
        # --------------------------------------------------------------------

        print(
            "\n"
            + "-" * 80,
            flush=True,
        )

        print(
            f"[{index}/{total_questions}] "
            f"Starting question {item.question_id}",
            flush=True,
        )

        print(
            f"Question: {item.question_text}",
            flush=True,
        )

        print(
            "-" * 80,
            flush=True,
        )

        # --------------------------------------------------------------------
        # Create per-question result
        # --------------------------------------------------------------------

        per = PerItemResult(
            question_id=item.question_id,
            question_text=item.question_text,
            is_answerable=item.is_answerable,
            ground_truth_answer=item.ground_truth_answer,
        )

        # --------------------------------------------------------------------
        # Gold evidence pages
        # --------------------------------------------------------------------

        gold_pages = [
            (
                evidence.source_doc_uid,
                evidence.source_page,
            )
            for evidence in item.gold_evidence
            if (
                evidence.source_doc_uid
                and evidence.source_page is not None
            )
        ]

        retrieved_pages: list[
            tuple[str, int]
        ] = []

        # --------------------------------------------------------------------
        # Langfuse trace
        # --------------------------------------------------------------------

        trace = None

        if lf:

            trace = lf.trace(
                name="benchmark_item",
                input={
                    "question_id": item.question_id,
                    "question": item.question_text,
                },
                metadata={
                    "langfuse_project": (
                        req.langfuse_project
                    ),
                },
            )

        # --------------------------------------------------------------------
        # Execute question
        # --------------------------------------------------------------------

        try:

            orch_url = (
                req.orchestrator_url
                or ORCHESTRATOR_URL_DEFAULT
            )

            print(
                f"[{index}/{total_questions}] "
                "Sending question to orchestrator...",
                flush=True,
            )

            pred = call_orchestrator(
                orchestrator_url=orch_url,
                question_text=item.question_text,
                question_id=item.question_id,
                debug=req.debug,
                trace_id=(
                    trace.id
                    if trace
                    else None
                ),
            )

            orchestrator_elapsed = (
                time.time()
                - question_start_time
            )

            print(
                f"[{index}/{total_questions}] "
                "Orchestrator finished "
                f"in {orchestrator_elapsed:.2f}s",
                flush=True,
            )

            # ----------------------------------------------------------------
            # Save complete orchestrator response
            # ----------------------------------------------------------------

            per.predicted_answer = pred

            per.predicted_answer_type = str(
                pred.get("answer_type")
            )

            # ----------------------------------------------------------------
            # Retrieval metrics
            # ----------------------------------------------------------------

            if req.debug:

                retrieved_pages = (
                    extract_retrieved_pages(
                        pred
                    )
                )

                print(
                    f"[{index}/{total_questions}] "
                    f"Retrieved pages: "
                    f"{len(retrieved_pages)}",
                    flush=True,
                )

            # ----------------------------------------------------------------
            # Extract predicted answer
            # ----------------------------------------------------------------

            answer_type = str(
                pred.get(
                    "answer_type",
                    "insufficient_evidence",
                )
            )

            params = pred.get(
                "params"
            ) or {}

            if answer_type in (
                "direct",
                "calculated",
            ):

                predicted_value = params.get(
                    "value"
                )

            elif answer_type == "multi_span":

                predicted_value = params.get(
                    "values"
                )

            else:

                predicted_value = None

            # ----------------------------------------------------------------
            # Score answer
            # ----------------------------------------------------------------

            if (
                item.is_answerable is False
                and answer_type
                == "insufficient_evidence"
            ):

                em = 1.0
                f1 = 1.0
                numeric_ok = None

            else:

                (
                    em,
                    f1,
                    numeric_ok,
                ) = score_prediction(
                    predicted_value,
                    item.ground_truth_answer,
                    item.scale,
                )

            per.em = em
            per.f1 = f1
            per.numeric_ok = numeric_ok

            ems.append(em)
            f1s.append(f1)

            if numeric_ok is not None:

                numeric_flags.append(
                    numeric_ok
                )

            # ----------------------------------------------------------------
            # Print answer metrics
            # ----------------------------------------------------------------

            print(
                f"[{index}/{total_questions}] "
                f"Answer type: {answer_type}",
                flush=True,
            )

            print(
                f"[{index}/{total_questions}] "
                f"Prediction: {predicted_value}",
                flush=True,
            )

            print(
                f"[{index}/{total_questions}] "
                f"EM: {em:.4f} | "
                f"F1: {f1:.4f} | "
                f"Numeric: {numeric_ok}",
                flush=True,
            )

            # ----------------------------------------------------------------
            # Page-level retrieval metrics
            # ----------------------------------------------------------------

            if retrieved_pages and gold_pages:

                metrics = (
                    compute_page_retrieval_metrics(
                        retrieved_pages,
                        gold_pages,
                        k=req.retrieval_k,
                    )
                )

                per.retrieval_hit = (
                    metrics["hit"]
                )

                per.retrieval_recall = (
                    metrics["recall"]
                )

                per.retrieval_precision = (
                    metrics["precision"]
                )

                per.retrieval_rr = (
                    metrics["rr"]
                )

                hits_k.append(
                    metrics["hit"]
                )

                recalls_k.append(
                    metrics["recall"]
                )

                precisions_k.append(
                    metrics["precision"]
                )

                rrs.append(
                    metrics["rr"]
                )

                print(
                    f"[{index}/{total_questions}] "
                    f"Retrieval @ {req.retrieval_k}: "
                    f"Hit={metrics['hit']:.0f} | "
                    f"Recall={metrics['recall']:.4f} | "
                    f"Precision={metrics['precision']:.4f} | "
                    f"RR={metrics['rr']:.4f}",
                    flush=True,
                )

            elif req.debug:

                print(
                    f"[{index}/{total_questions}] "
                    "Retrieval metrics unavailable "
                    "(missing retrieved or gold pages)",
                    flush=True,
                )

            # ----------------------------------------------------------------
            # Langfuse scores
            # ----------------------------------------------------------------

            if trace:

                trace.score(
                    name="em",
                    value=em,
                )

                trace.score(
                    name="f1",
                    value=f1,
                )

                if numeric_ok is not None:

                    trace.score(
                        name="numeric_ok",
                        value=(
                            1.0
                            if numeric_ok
                            else 0.0
                        ),
                    )

        except Exception as e:

            per.error = str(e)

            print(
                f"[{index}/{total_questions}] "
                f"ERROR: {e}",
                flush=True,
            )

            if trace:

                trace.score(
                    name="error",
                    value=1.0,
                )

        finally:

            if trace:

                trace.update(
                    output={
                        "result": (
                            per.model_dump()
                        )
                    }
                )

        # --------------------------------------------------------------------
        # Save result in memory
        # --------------------------------------------------------------------

        results.append(per)

        # --------------------------------------------------------------------
        # Question completion
        # --------------------------------------------------------------------

        question_elapsed = (
            time.time()
            - question_start_time
        )

        benchmark_elapsed = (
            time.time()
            - benchmark_start_time
        )

        completed = index

        remaining = (
            total_questions
            - completed
        )

        average_time = (
            benchmark_elapsed / completed
        )

        estimated_remaining = (
            average_time * remaining
        )

        print(
            f"[{index}/{total_questions}] "
            f"Finished question {item.question_id} "
            f"in {question_elapsed:.2f}s",
            flush=True,
        )

        if remaining > 0:

            print(
                f"[{index}/{total_questions}] "
                f"Estimated remaining time: "
                f"{estimated_remaining:.1f}s",
                flush=True,
            )

        # --------------------------------------------------------------------
        # Optional delay
        # --------------------------------------------------------------------

        if req.delay_seconds > 0:

            print(
                f"[{index}/{total_questions}] "
                f"Waiting {req.delay_seconds:.1f}s...",
                flush=True,
            )

            time.sleep(
                req.delay_seconds
            )

    # =========================================================================
    # Calculate final metrics
    # =========================================================================

    avg_em = (
        sum(ems) / len(ems)
        if ems
        else None
    )

    avg_f1 = (
        sum(f1s) / len(f1s)
        if f1s
        else None
    )

    numeric_acc = (
        sum(
            1
            for value in numeric_flags
            if value
        )
        / len(numeric_flags)
        if numeric_flags
        else None
    )

    avg_hit_at_k = (
        sum(hits_k) / len(hits_k)
        if hits_k
        else None
    )

    avg_recall_at_k = (
        sum(recalls_k) / len(recalls_k)
        if recalls_k
        else None
    )

    avg_precision_at_k = (
        sum(precisions_k)
        / len(precisions_k)
        if precisions_k
        else None
    )

    mrr_at_k = (
        sum(rrs) / len(rrs)
        if rrs
        else None
    )

    # =========================================================================
    # Create final response
    # =========================================================================

    response = RunBenchmarkResponse(
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

    # =========================================================================
    # Save benchmark results
    # =========================================================================

    # Automatically creates the directory if it does not exist.
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            response.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # Flush Langfuse
    # =========================================================================

    if lf:
        lf.flush()

    # =========================================================================
    # Final terminal summary
    # =========================================================================

    total_elapsed = (
        time.time()
        - benchmark_start_time
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "LEDGER BENCHMARK FINISHED",
        flush=True,
    )

    print(
        f"Questions: {len(items)}",
        flush=True,
    )

    print(
        f"Scored: {len(ems)}",
        flush=True,
    )

    print(
        f"Exact Match: "
        f"{avg_em:.4f}"
        if avg_em is not None
        else "Exact Match: N/A",
        flush=True,
    )

    print(
        f"F1: "
        f"{avg_f1:.4f}"
        if avg_f1 is not None
        else "F1: N/A",
        flush=True,
    )

    print(
        f"Numeric Accuracy: "
        f"{numeric_acc:.4f}"
        if numeric_acc is not None
        else "Numeric Accuracy: N/A",
        flush=True,
    )

    print(
        f"Hit@{req.retrieval_k}: "
        f"{avg_hit_at_k:.4f}"
        if avg_hit_at_k is not None
        else f"Hit@{req.retrieval_k}: N/A",
        flush=True,
    )

    print(
        f"Recall@{req.retrieval_k}: "
        f"{avg_recall_at_k:.4f}"
        if avg_recall_at_k is not None
        else f"Recall@{req.retrieval_k}: N/A",
        flush=True,
    )

    print(
        f"Precision@{req.retrieval_k}: "
        f"{avg_precision_at_k:.4f}"
        if avg_precision_at_k is not None
        else f"Precision@{req.retrieval_k}: N/A",
        flush=True,
    )

    print(
        f"MRR@{req.retrieval_k}: "
        f"{mrr_at_k:.4f}"
        if mrr_at_k is not None
        else f"MRR@{req.retrieval_k}: N/A",
        flush=True,
    )

    print(
        f"Total time: {total_elapsed:.2f}s",
        flush=True,
    )

    print(
        f"Results saved to: {RESULTS_FILE}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    return response

"""
curl -sS -X POST http://localhost:8005/run_benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "orchestrator_url": "http://orchestrator:8000",
    "start": 90,
    "end": 100,
    "debug": true,
    "retrieval_k": 5,
    "delay_seconds": 0
  }'



docker compose logs -f eval-service
"""