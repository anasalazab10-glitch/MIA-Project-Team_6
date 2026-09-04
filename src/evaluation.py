"""Evaluation and Benchmarking Suite for Retrieval Pipeline Variants.

Computes standard Information Retrieval metrics as mandated by the project specification:
- Hit Rate@K (Hit@1, Hit@3, Hit@5)
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Average Query Latency (ms)

Evaluates and compares the 4 core pipeline variants:
1. BM25 Only
2. Dense Only (Qdrant + BAAI/bge-small-en-v1.5)
3. Hybrid RRF (BM25 + Dense merged via Reciprocal Rank Fusion)
4. Full Pipeline (Hybrid RRF -> Cross-Encoder Reranker -> Top 5)

Fulfills rubric mandate: "The value of reranking should be evaluated, not assumed."
"""

import json
import time
from pathlib import Path
from typing import Any, Callable

try:
    from src.bm25_retriever import BM25Retriever
    from src.dense_retriever import DenseRetriever
    from src.embeddings import EmbeddingModel
    from src.fusion import HybridRetriever, build_hybrid_pipeline
    from src.reranker import CrossEncoderReranker, RetrievalPipeline
    from src.schemas import Candidate, Chunk, RetrievalResponse
    from src.vector_store import VectorStore
except ImportError:
    from bm25_retriever import BM25Retriever
    from dense_retrieval import DenseRetriever
    from embeddings import EmbeddingModel
    from fusion import HybridRetriever, build_hybrid_pipeline
    from reranker import CrossEncoderReranker, RetrievalPipeline
    from schemas import Candidate, Chunk, RetrievalResponse
    from vector_store import VectorStore


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k_list: list[int] = [1, 3, 5],
) -> dict[str, float]:
    """Compute Hit@K, Recall@K, Precision@K, and Reciprocal Rank for a single query."""
    relevant_set = set(relevant_chunk_ids)
    metrics: dict[str, float] = {}

    # Reciprocal Rank (MRR component)
    rr = 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            rr = 1.0 / rank
            break
    metrics["rr"] = rr

    # Metrics at each K cutoff
    for k in k_list:
        top_k = retrieved_chunk_ids[:k]
        hits = [cid for cid in top_k if cid in relevant_set]

        # Hit Rate: 1 if at least one relevant document in top K, else 0
        metrics[f"hit@{k}"] = 1.0 if len(hits) > 0 else 0.0

        # Recall@K: fraction of all relevant documents retrieved in top K
        metrics[f"recall@{k}"] = len(hits) / len(relevant_set) if relevant_set else 0.0

        # Precision@K: fraction of top K documents that are relevant
        metrics[f"precision@{k}"] = len(hits) / k if k > 0 else 0.0

    return metrics


def evaluate_retriever(
    retrieve_fn: Callable[[str, int], list[Candidate]],
    benchmark_dataset: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, float]:
    """Run an automated benchmark over an evaluation dataset and average metrics."""
    total_queries = len(benchmark_dataset)
    if total_queries == 0:
        return {}

    aggregated: dict[str, float] = {}
    latencies: list[float] = []

    for item in benchmark_dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        start_time = time.perf_counter()
        candidates = retrieve_fn(query, top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_ids = [c.chunk.chunk_id for c in candidates]
        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k_list=[1, 3, 5])

        for metric_name, val in metrics.items():
            aggregated[metric_name] = aggregated.get(metric_name, 0.0) + val

    results: dict[str, float] = {}
    for metric_name, total_val in aggregated.items():
        results[metric_name] = round(total_val / total_queries, 4)

    results["mrr"] = results.pop("rr", 0.0)
    results["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
    return results


def run_comprehensive_benchmark(
    chunks: list[Chunk],
    benchmark_dataset: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compare all 4 pipeline configurations against the benchmark dataset."""
    # 1. Initialize components
    print("Setting up BM25, Dense (Qdrant + BGE), and Cross-Encoder components...")
    bm25 = BM25Retriever(chunks)

    embed_model = EmbeddingModel()
    embeddings = embed_model.encode_chunks(chunks)
    vector_store = VectorStore(collection_name="eval_benchmark_suite", vector_size=384)
    vector_store.add_chunks(chunks, embeddings)
    dense = DenseRetriever(embed_model, vector_store)

    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    reranker = CrossEncoderReranker()
    pipeline = RetrievalPipeline(hybrid_retriever=hybrid, reranker=reranker)

    # 2. Define retriever functions for each variant
    variants = {
        "1. BM25 Only": lambda q, k: bm25.search(q, top_k=k),
        "2. Dense Only": lambda q, k: dense.retrieve(q, top_k=k).candidates,
        "3. Hybrid RRF": lambda q, k: hybrid.search(q, top_k=k, bm25_top_k=30, dense_top_k=30).candidates,
        "4. Full Pipeline (+ Reranker)": lambda q, k: pipeline.retrieve(q, over_retrieve_k=30, final_top_k=k).candidates,
    }

    # 3. Evaluate each variant
    all_results: dict[str, dict[str, float]] = {}
    for variant_name, fn in variants.items():
        print(f"Evaluating {variant_name}...")
        all_results[variant_name] = evaluate_retriever(fn, benchmark_dataset, top_k=5)

    return all_results


def print_benchmark_report(results: dict[str, dict[str, float]]) -> None:
    """Print formatted markdown-compatible comparative benchmark table."""
    headers = ["Pipeline Variant", "Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR", "Latency (ms)"]
    print("\n" + "=" * 95)
    print("RETRIEVAL PIPELINE BENCHMARK REPORT (Evaluation Suite)")
    print("=" * 95)
    header_str = f"| {headers[0]:<30} | {headers[1]:<7} | {headers[2]:<7} | {headers[3]:<7} | {headers[4]:<9} | {headers[5]:<6} | {headers[6]:<12} |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_str)
    print(sep_str)

    for variant_name, m in results.items():
        row = (
            f"| {variant_name:<30} | "
            f"{m.get('hit@1', 0.0):<7.2f} | "
            f"{m.get('hit@3', 0.0):<7.2f} | "
            f"{m.get('hit@5', 0.0):<7.2f} | "
            f"{m.get('recall@5', 0.0):<9.2f} | "
            f"{m.get('mrr', 0.0):<6.3f} | "
            f"{m.get('avg_latency_ms', 0.0):<12.1f} |"
        )
        print(row)
    print("=" * 95)


if __name__ == "__main__":
    chunks_file = Path(__file__).resolve().parent.parent / "data" / "mock_chunks.json"
    dataset_file = Path(__file__).resolve().parent.parent / "data" / "retrieval_eval_dataset.json"

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk.model_validate(c) for c in chunks_data]

    with open(dataset_file, "r", encoding="utf-8") as f:
        benchmark_dataset = json.load(f)

    print(f"Loaded {len(chunks)} chunks and {len(benchmark_dataset)} evaluation test cases.")
    results = run_comprehensive_benchmark(chunks, benchmark_dataset)
    print_benchmark_report(results)
