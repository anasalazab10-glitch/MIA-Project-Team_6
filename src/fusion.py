"""Reciprocal Rank Fusion (RRF) & Hybrid Retrieval Pipeline.

Merges ranked candidate lists from BM25 keyword search and Dense vector search
(Qdrant + BAAI/bge-small-en-v1.5) into a unified, high-recall candidate pool
for downstream cross-encoder reranking.

Formula:
    RRF_score(d) = \\sum_{m \\in M} \\frac{w_m}{k + rank_m(d)}
"""

import json
from pathlib import Path
from typing import Any

try:
    from src.bm25_retriever import BM25Retriever
    from src.dense_retrieval import DenseRetriever
    from src.embeddings import EmbeddingModel
    from src.schemas import Candidate, Chunk, RetrievalMethod, RetrievalResponse
    from src.vector_store import VectorStore
except ImportError:
    from bm25_retriever import BM25Retriever
    from dense_retrieval import DenseRetriever
    from embeddings import EmbeddingModel
    from schemas import Candidate, Chunk, RetrievalMethod, RetrievalResponse
    from vector_store import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Candidate]] | list[list[Candidate]],
    k: int = 60,
    weights: dict[str, float] | list[float] | None = None,
    top_k: int = 30,
) -> list[Candidate]:
    """Fuse multiple ranked Candidate lists into a single unified ranked list using RRF.

    Args:
        ranked_lists: Dictionary mapping method name to candidate list
                      (e.g., {"bm25": bm25_results, "dense": dense_results})
                      or a list of candidate lists.
        k: Smoothing constant to control the weight given to lower-ranked items (default 60).
        weights: Optional weights for each retriever (default 1.0 for each).
        top_k: Maximum number of fused candidates to return (default 30 for reranker stage).

    Returns:
        List of merged Candidate objects sorted descending by fused RRF score.
    """
    normalized_lists: dict[str, list[Candidate]] = {}
    normalized_weights: dict[str, float] = {}

    if isinstance(ranked_lists, dict):
        normalized_lists = ranked_lists
        if isinstance(weights, dict):
            normalized_weights = {name: weights.get(name, 1.0) for name in ranked_lists}
        elif isinstance(weights, list):
            for i, name in enumerate(ranked_lists):
                normalized_weights[name] = weights[i] if i < len(weights) else 1.0
        else:
            normalized_weights = {name: 1.0 for name in ranked_lists}
    else:
        for idx, cand_list in enumerate(ranked_lists):
            name = f"retriever_{idx}"
            normalized_lists[name] = cand_list
            if isinstance(weights, list) and idx < len(weights):
                normalized_weights[name] = weights[idx]
            else:
                normalized_weights[name] = 1.0

    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, Chunk] = {}
    source_scores: dict[str, dict[str, float]] = {}
    source_ranks: dict[str, dict[str, int]] = {}

    for method_name, candidates in normalized_lists.items():
        weight = normalized_weights.get(method_name, 1.0)

        for rank, candidate in enumerate(candidates, start=1):
            chunk = candidate.chunk
            chunk_id = chunk.chunk_id

            if chunk_id not in chunk_store:
                chunk_store[chunk_id] = chunk
                source_scores[chunk_id] = {}
                source_ranks[chunk_id] = {}

            # RRF calculation: weight / (k + rank)
            component = weight / (k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + component

            source_scores[chunk_id][method_name] = candidate.score
            source_ranks[chunk_id][f"{method_name}_rank"] = rank

            if candidate.scores:
                for k_score, v_score in candidate.scores.items():
                    if k_score not in source_scores[chunk_id]:
                        source_scores[chunk_id][k_score] = v_score

    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    fused_candidates: list[Candidate] = []
    for rank, (chunk_id, score) in enumerate(sorted_chunks[:top_k], start=1):
        chunk = chunk_store[chunk_id]

        scores_dict = dict(source_scores[chunk_id])
        scores_dict["rrf"] = round(score, 6)

        # Attach source ranks to chunk metadata for complete tracing in Langfuse
        metadata_copy = dict(chunk.metadata)
        metadata_copy.update(source_ranks[chunk_id])

        fused_chunk = chunk.model_copy(update={"metadata": metadata_copy})

        fused_candidates.append(
            Candidate(
                chunk=fused_chunk,
                score=round(score, 6),
                retrieval_method=RetrievalMethod.RRF,
                rank=rank,
                initial_rank=rank,
                scores=scores_dict,
            )
        )

    return fused_candidates


class RRFFusion:
    """Configurable Reciprocal Rank Fusion stage."""

    def __init__(
        self,
        k: int = 60,
        weights: dict[str, float] | None = None,
        default_top_k: int = 30,
    ) -> None:
        self.k = k
        self.weights = weights or {"bm25": 1.0, "dense": 1.0}
        self.default_top_k = default_top_k

    def fuse(
        self,
        ranked_lists: dict[str, list[Candidate]] | list[list[Candidate]],
        top_k: int | None = None,
    ) -> list[Candidate]:
        k_val = top_k if top_k is not None else self.default_top_k
        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=self.k,
            weights=self.weights,
            top_k=k_val,
        )


class HybridRetriever:
    """Integrated Hybrid Retriever orchestrating BM25, Dense Retrieval, and RRF Fusion.
    
    Combines lexical precision (BM25) with semantic vector recall (Dense/Qdrant)
    to return top candidates for the downstream Cross-Encoder reranker.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion: RRFFusion | None = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.fusion = fusion or RRFFusion(k=60, weights={"bm25": 1.0, "dense": 1.0})

    def search(
        self,
        query: str,
        top_k: int = 30,
        bm25_top_k: int = 30,
        dense_top_k: int = 30,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Run hybrid retrieval: executes BM25 and Dense search, then fuses with RRF."""
        # 1. Lexical retrieval via BM25
        bm25_candidates = self.bm25_retriever.search(
            query=query,
            top_k=bm25_top_k,
            metadata_filter=metadata_filter,
        )

        # 2. Semantic vector retrieval via DenseRetriever (Qdrant)
        dense_response = self.dense_retriever.retrieve(
            query=query,
            top_k=dense_top_k,
        )
        dense_candidates = dense_response.candidates

        # 3. Fuse candidate sets using Reciprocal Rank Fusion
        fused_candidates = self.fusion.fuse(
            ranked_lists={
                "bm25": bm25_candidates,
                "dense": dense_candidates,
            },
            top_k=top_k,
        )

        return RetrievalResponse(
            query=query,
            retrieval_method=RetrievalMethod.RRF,
            candidates=fused_candidates,
        )


def build_hybrid_pipeline(
    chunks: list[Chunk],
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    collection_name: str = "hybrid_ledger_chunks",
    vector_size: int = 384,
) -> HybridRetriever:
    """Factory function to instantiate and index both BM25 and Dense retrievers."""
    # 1. Build BM25 index
    bm25 = BM25Retriever(chunks)

    # 2. Build Dense vector index
    embedding_model = EmbeddingModel(model_name=embedding_model_name)
    embeddings = embedding_model.encode_chunks(chunks)

    vector_store = VectorStore(
        collection_name=collection_name,
        vector_size=vector_size,
    )
    vector_store.add_chunks(chunks, embeddings)

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)


# ---------------------------------------------------------------------------
# End-to-End Hybrid Search Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock_path = Path(__file__).resolve().parent.parent / "data" / "mock_chunks.json"
    with open(mock_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [Chunk.model_validate(item) for item in data]
    print(f"Loaded {len(chunks)} chunks for end-to-end Hybrid RRF test.\n")

    print("Building Hybrid Retriever (BM25 + Qdrant Dense + BGE embeddings)...")
    hybrid = build_hybrid_pipeline(chunks)
    print("Hybrid Retriever successfully built!\n")

    test_queries = [
        "How much R&D did the company spend in 2023?",
        "What was the operating income and net profit in 2024?",
        "Asia-Pacific revenue growth rate",
    ]

    for q in test_queries:
        print("=" * 70)
        print(f"Query: '{q}'")
        print("=" * 70)
        response = hybrid.search(q, top_k=5)

        for cand in response.candidates:
            b_rank = cand.chunk.metadata.get("bm25_rank", "-")
            d_rank = cand.chunk.metadata.get("dense_rank", "-")
            bm25_s = cand.scores.get("bm25", "-")
            dense_s = cand.scores.get("dense", "-")
            print(
                f"[Rank {cand.rank}] RRF Score: {cand.score:.6f} | "
                f"ID: {cand.chunk.chunk_id} | "
                f"BM25 Rank: {b_rank} (Score: {bm25_s}) | "
                f"Dense Rank: {d_rank} (Score: {dense_s}) | "
                f"Section: '{cand.chunk.section}'"
            )
        print()
