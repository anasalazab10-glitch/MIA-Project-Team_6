"""
Evaluation and Benchmarking for the Ledger Retrieval Pipeline.

The evaluation measures every stage independently:

1. Dense Retrieval  -> Top 30
2. BM25 Retrieval   -> Top 30
3. Hybrid RRF       -> Top 30
4. Cross-Encoder    -> Final Top 5

All stages use the SAME chunks loaded from the existing
Qdrant collection used by the retrieval API.

it connects to the existing Qdrant collection,
loads the stored chunks, builds BM25 from those chunks,
and uses the existing Qdrant vectors for Dense Retrieval.

retrieval_eval_dataset.json contains the ground-truth
relevant chunk IDs for each query.

Metrics:
- Hit@K
- Recall@K
- Precision@K
- MRR
- Average latency
"""

import json
import time
from pathlib import Path
from typing import Any, Callable

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.embeddings import EmbeddingModel
from src.fusion import HybridRetriever, RRFFusion
from src.reranker import CrossEncoderReranker, RetrievalPipeline
from src.schemas import Candidate, Chunk
from src.vector_store import VectorStore


DENSE_TOP_K = 30
BM25_TOP_K = 30
HYBRID_TOP_K = 30
FINAL_TOP_K = 5

QDRANT_COLLECTION = "ledger_chunks"
VECTOR_SIZE = 384
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Metrics

def compute_metrics(retrieved_chunk_ids: list[str],relevant_chunk_ids: list[str],k: int,) -> dict[str, float]:
    """
    Compute Hit@K, Recall@K, Precision@K and
    Reciprocal Rank for a single query.
    """

    relevant_set = set(relevant_chunk_ids)
    retrieved_at_k = retrieved_chunk_ids[:k]

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_at_k
        if chunk_id in relevant_set
    ]

    number_relevant = len(relevant_retrieved)

    # Hit@K
    hit = 1.0 if number_relevant > 0 else 0.0


    # Recall@K

    recall = (
        number_relevant / len(relevant_set)
        if relevant_set
        else 0.0
    )


    # Precision@K
    precision = (
        number_relevant / k
        if k > 0
        else 0.0
    )


    # Reciprocal Rank

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "rr": reciprocal_rank,
    }



# Evaluate One Stage

def evaluate_stage(stage_name: str,retrieve_fn: Callable[[str, int], list[Candidate]],benchmark_dataset: list[dict[str, Any]],k: int,) -> dict[str, float]:
    """
    Evaluate one retrieval stage over the complete
    benchmark dataset.
    """

    print()
    print("=" * 90)
    print(stage_name)
    print("=" * 90)

    totals = {
        "hit": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "rr": 0.0,
    }

    latencies = []

    for item in benchmark_dataset:

        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        # Measure retrieval latency
        start = time.perf_counter()

        candidates = retrieve_fn(query, k)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

        # Extract retrieved chunk IDs
        retrieved_ids = [
            candidate.chunk.chunk_id
            for candidate in candidates
        ]


        # Compute metrics

        metrics = compute_metrics(
            retrieved_ids,
            relevant_ids,
            k,
        )

        for metric_name, value in metrics.items():
            totals[metric_name] += value

  
        # Print query result
       
        print()
        print(f"Query: {query}")

        print(
            f"Relevant chunks: {relevant_ids}"
        )

        print(
            f"Retrieved Top {k}: "
            f"{retrieved_ids[:k]}"
        )

        print(
            f"Hit@{k}: {metrics['hit']:.2f} | "
            f"Recall@{k}: {metrics['recall']:.2f} | "
            f"Precision@{k}: {metrics['precision']:.2f} | "
            f"RR: {metrics['rr']:.3f}"
        )

        print(
            f"Latency: {elapsed_ms:.2f} ms"
        )


    # Calculate averages
    number_queries = len(benchmark_dataset)

    if number_queries == 0:
        return {}

    results = {
        "hit": totals["hit"] / number_queries,
        "recall": totals["recall"] / number_queries,
        "precision": totals["precision"] / number_queries,
        "mrr": totals["rr"] / number_queries,
        "avg_latency_ms": (
            sum(latencies) / len(latencies)
        ),
    }


    # Stage Summary
   
    print()
    print("-" * 90)
    print(f"{stage_name} - SUMMARY")
    print("-" * 90)

    print(
        f"Hit@{k}:          "
        f"{results['hit']:.4f}"
    )

    print(
        f"Recall@{k}:       "
        f"{results['recall']:.4f}"
    )

    print(
        f"Precision@{k}:    "
        f"{results['precision']:.4f}"
    )

    print(
        f"MRR:              "
        f"{results['mrr']:.4f}"
    )

    print(
        f"Avg latency:      "
        f"{results['avg_latency_ms']:.2f} ms"
    )

    return results


# Benchmark

def run_benchmark(chunks: list[Chunk],benchmark_dataset: list[dict[str, Any]],):
    """
    Build the retrieval pipeline using the existing Qdrant
    collection and evaluate every retrieval stage.
    The chunks are already stored in Qdrant.
    """

    print()
    print("=" * 90)
    print("LEDGER RETRIEVAL PIPELINE EVALUATION")
    print("=" * 90)

    print(
        f"Number of chunks loaded from Qdrant: "
        f"{len(chunks)}"
    )

    print(
        f"Number of queries: "
        f"{len(benchmark_dataset)}"
    )

  
    # Connect to Existing Qdrant
    print()
    print("Connecting to existing Qdrant collection...")

    vector_store = VectorStore(
        collection_name=QDRANT_COLLECTION,
        vector_size=VECTOR_SIZE,
    )

    print("Connected to Qdrant.")

    print(
        f"Collection: "
        f"{vector_store.collection_name}"
    )

    # Build BM25 from Qdrant Chunks
    print()
    print("Building BM25 index from Qdrant chunks...")

    bm25 = BM25Retriever(chunks)

    print("BM25 ready.")

    
    # Load Embedding Model

    print()
    print("Loading embedding model...")

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded.")

    # Dense Retriever

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    print("Dense retriever ready.")

 
    # Hybrid RRF
    print()
    print("Building Hybrid RRF...")

    fusion = RRFFusion(
        k=60,
        weights={
            "bm25": 1.0,
            "dense": 1.0,
        },
        default_top_k=HYBRID_TOP_K,
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        fusion=fusion,
    )

    print("Hybrid retriever ready.")

    # Cross-Encoder

    print()
    print("Loading Cross-Encoder...")

    reranker = CrossEncoderReranker()

    print("Cross-Encoder ready.")

    # Full Retrieval Pipeline

    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        over_retrieve_k=HYBRID_TOP_K,
        final_top_k=FINAL_TOP_K,
    )

    print("Full retrieval pipeline ready.")

 
    # 1. Dense Top 30

    dense_results = evaluate_stage(
        stage_name="1. DENSE RETRIEVAL - TOP 30",

        retrieve_fn=lambda query, k: (
            dense
            .retrieve(
                query,
                top_k=k,
            )
            .candidates
        ),

        benchmark_dataset=benchmark_dataset,

        k=DENSE_TOP_K,
    )

 
    # 2. BM25 Top 30

    bm25_results = evaluate_stage(
        stage_name="2. BM25 RETRIEVAL - TOP 30",

        retrieve_fn=lambda query, k: (
            bm25.search(
                query,
                top_k=k,
            )
        ),

        benchmark_dataset=benchmark_dataset,

        k=BM25_TOP_K,
    )


    # 3. Hybrid RRF Top 30
    hybrid_results = evaluate_stage(
        stage_name="3. HYBRID RRF - TOP 30",

        retrieve_fn=lambda query, k: (
            hybrid
            .search(
                query,
                top_k=k,
                bm25_top_k=BM25_TOP_K,
                dense_top_k=DENSE_TOP_K,
            )
            .candidates
        ),

        benchmark_dataset=benchmark_dataset,

        k=HYBRID_TOP_K,
    )

    # 4. Hybrid + Cross-Encoder Top 5

    reranked_results = evaluate_stage(
        stage_name=(
            "4. HYBRID + CROSS-ENCODER "
            "RERANKER - TOP 5"
        ),

        retrieve_fn=lambda query, k: (
            pipeline
            .retrieve(
                query,
                over_retrieve_k=HYBRID_TOP_K,
                final_top_k=k,
            )
            .candidates
        ),

        benchmark_dataset=benchmark_dataset,

        k=FINAL_TOP_K,
    )


    print()
    print()
    print("=" * 115)
    print("FINAL PIPELINE COMPARISON")
    print("=" * 115)

    print(
        f"{'Stage':<45}"
        f"{'Hit':<10}"
        f"{'Recall':<10}"
        f"{'Precision':<12}"
        f"{'MRR':<10}"
        f"{'Latency (ms)':<15}"
    )

    print("-" * 115)

    print(
        f"{'Dense Top 30':<45}"
        f"{dense_results['hit']:<10.4f}"
        f"{dense_results['recall']:<10.4f}"
        f"{dense_results['precision']:<12.4f}"
        f"{dense_results['mrr']:<10.4f}"
        f"{dense_results['avg_latency_ms']:<15.2f}"
    )

    print(
        f"{'BM25 Top 30':<45}"
        f"{bm25_results['hit']:<10.4f}"
        f"{bm25_results['recall']:<10.4f}"
        f"{bm25_results['precision']:<12.4f}"
        f"{bm25_results['mrr']:<10.4f}"
        f"{bm25_results['avg_latency_ms']:<15.2f}"
    )

    print(
        f"{'Hybrid RRF Top 30':<45}"
        f"{hybrid_results['hit']:<10.4f}"
        f"{hybrid_results['recall']:<10.4f}"
        f"{hybrid_results['precision']:<12.4f}"
        f"{hybrid_results['mrr']:<10.4f}"
        f"{hybrid_results['avg_latency_ms']:<15.2f}"
    )

    print(
        f"{'Hybrid + Reranker Top 5':<45}"
        f"{reranked_results['hit']:<10.4f}"
        f"{reranked_results['recall']:<10.4f}"
        f"{reranked_results['precision']:<12.4f}"
        f"{reranked_results['mrr']:<10.4f}"
        f"{reranked_results['avg_latency_ms']:<15.2f}"
    )

    print("=" * 115)

    return {
        "dense_top_30": dense_results,
        "bm25_top_30": bm25_results,
        "hybrid_top_30": hybrid_results,
        "reranked_top_5": reranked_results,
    }


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    # Load Ground-Truth Evaluation Dataset
 
    evaluation_file = (
        Path(__file__).resolve().parent
        / "retrieval_eval_dataset.json"
    )

    with open(
        evaluation_file,
        "r",
        encoding="utf-8",
    ) as f:
        benchmark_dataset = json.load(f)

    print(
        f"Loaded {len(benchmark_dataset)} "
        f"evaluation queries."
    )

    # Connect to Existing Qdrant
    print()
    print("Loading chunks from existing Qdrant...")

    vector_store = VectorStore(
        collection_name=QDRANT_COLLECTION,
        vector_size=VECTOR_SIZE,
    )

    chunks = vector_store.get_all_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks found in Qdrant. "
            "Run the indexing pipeline first."
        )

    print(
        f"Loaded {len(chunks)} chunks from "
        f"Qdrant collection "
        f"'{QDRANT_COLLECTION}'."
    )

    available_chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    missing_ids = set()

    for item in benchmark_dataset:

        for chunk_id in item["relevant_chunk_ids"]:

            if chunk_id not in available_chunk_ids:
                missing_ids.add(chunk_id)

    if missing_ids:

        print()
        print("=" * 90)
        print(
            "ERROR: UNKNOWN CHUNK IDs "
            "IN EVALUATION DATASET"
        )
        print("=" * 90)

        print(
            "The following ground-truth IDs "
            "do not exist in Qdrant:"
        )

        for chunk_id in sorted(missing_ids):
            print(f"  - {chunk_id}")

        print()
        print("Available chunk IDs:")

        for chunk_id in sorted(
            available_chunk_ids
        ):
            print(f"  - {chunk_id}")

        raise ValueError(
            "retrieval_eval_dataset.json contains "
            "chunk IDs that are not present in "
            "the existing Qdrant collection."
        )

    print()
    print(
        "All ground-truth chunk IDs "
        "were found in Qdrant."
    )

    # Run Benchmark
    results = run_benchmark(
        chunks=chunks,
        benchmark_dataset=benchmark_dataset,
    )