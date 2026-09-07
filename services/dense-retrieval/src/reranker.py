"""Cross-Encoder Reranker for Financial Document Retrieval.

Over-retrieves candidate chunks from the hybrid RRF stage (e.g., top 30) and
scores (query, chunk) pairs jointly with full cross-attention to produce the
final top 5 candidates for the reasoning agent.

Project Requirement (Specification Page 3):
    "Candidates should be over-retrieved and reranked before being handed to the
    agent (e.g. top 30 -> reranker -> top 5); the value of reranking should be
    evaluated, not assumed."
"""

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path regardless of working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sentence_transformers import CrossEncoder

from src.fusion import HybridRetriever, build_hybrid_pipeline
from src.schemas import Candidate, Chunk, RetrievalMethod, RetrievalResponse


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OVER_RETRIEVE_K = 30
DEFAULT_FINAL_TOP_K = 5


class CrossEncoderReranker:
    """Resource-efficient Cross-Encoder reranker using cross-attention."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name=model_name,
            max_length=max_length,
            device=device,
        )

    def _prepare_document_text(self, chunk: Chunk) -> str:
        """Construct rich text representation including section context for reranking."""
        section_prefix = f"Section: {chunk.section}\n" if chunk.section else ""
        return f"{section_prefix}{chunk.to_text()}"

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> list[Candidate]:
        """Rerank an over-retrieved candidate list using Cross-Encoder joint scoring.

        Args:
            query: The user question or search query.
            candidates: Over-retrieved candidates (e.g., top 30 from RRF or Dense).
            top_k: Number of final top candidates to return (default 5).

        Returns:
            Ranked list of top_k Candidate objects with updated scores and ranks.
        """
        if not candidates:
            return []

        # Prepare (query, document) pairs for joint cross-attention scoring
        pairs = [
            [query, self._prepare_document_text(candidate.chunk)]
            for candidate in candidates
        ]

        # Predict cross-encoder relevance logits
        raw_scores = self.model.predict(pairs)

        # Handle scalar output if only 1 pair
        if hasattr(raw_scores, "tolist"):
            scores_list = raw_scores.tolist()
        elif isinstance(raw_scores, (list, tuple)):
            scores_list = list(raw_scores)
        else:
            scores_list = [float(raw_scores)]

        # Pair candidates with their rerank scores
        scored_candidates: list[tuple[Candidate, float]] = []
        for candidate, rerank_score in zip(candidates, scores_list):
            scored_candidates.append((candidate, float(rerank_score)))

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        # Build updated Candidate models preserving history
        reranked: list[Candidate] = []
        for rank, (candidate, rerank_score) in enumerate(scored_candidates[:top_k], start=1):
            scores_dict = dict(candidate.scores)
            scores_dict["reranker"] = round(rerank_score, 4)

            # Preserve initial rank from the previous stage (e.g. RRF rank)
            initial_rank = candidate.initial_rank or candidate.rank

            # Merge previous rank into metadata for Langfuse evaluation
            metadata_copy = dict(candidate.chunk.metadata)
            metadata_copy["pre_rerank_rank"] = candidate.rank
            metadata_copy["pre_rerank_score"] = candidate.score

            updated_chunk = candidate.chunk.model_copy(update={"metadata": metadata_copy})

            reranked.append(
                Candidate(
                    chunk=updated_chunk,
                    score=round(rerank_score, 4),
                    retrieval_method=RetrievalMethod.RERANKER,
                    rank=rank,
                    initial_rank=initial_rank,
                    scores=scores_dict,
                )
            )

        return reranked

    def rerank_response(
        self,
        response: RetrievalResponse,
        top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> RetrievalResponse:
        """Convenience method to rerank a RetrievalResponse directly."""
        reranked_candidates = self.rerank(
            query=response.query,
            candidates=response.candidates,
            top_k=top_k,
        )
        return RetrievalResponse(
            query=response.query,
            retrieval_method=RetrievalMethod.RERANKER,
            candidates=reranked_candidates,
        )


class RetrievalPipeline:
    """End-to-End Retrieval Pipeline: Hybrid Search (BM25 + Dense -> RRF) -> Reranker -> Top 5."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        over_retrieve_k: int = DEFAULT_OVER_RETRIEVE_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.over_retrieve_k = over_retrieve_k
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        over_retrieve_k: int | None = None,
        final_top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Execute full pipeline: over-retrieve candidates and rerank down to top 5."""
        k_over = over_retrieve_k if over_retrieve_k is not None else self.over_retrieve_k
        k_final = final_top_k if final_top_k is not None else self.final_top_k

        # 1. Over-retrieve candidates via Hybrid RRF (top 30)
        hybrid_response = self.hybrid_retriever.search(
            query=query,
            top_k=k_over,
            bm25_top_k=k_over,
            dense_top_k=k_over,
            metadata_filter=metadata_filter,
        )

        # 2. Rerank down to final top 5 via Cross-Encoder
        return self.reranker.rerank_response(
            response=hybrid_response,
            top_k=k_final,
        )


# ---------------------------------------------------------------------------
# Quick Verification / Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock_path = Path(__file__).resolve().parent.parent / "data" / "mock_chunks.json"
    with open(mock_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [Chunk.model_validate(item) for item in data]
    print(f"Loaded {len(chunks)} chunks for full pipeline test.\n")

    print("Building Hybrid Retriever and Cross-Encoder Reranker...")
    hybrid = build_hybrid_pipeline(chunks)
    pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid,
        over_retrieve_k=30,
        final_top_k=5,
    )
    print("Pipeline ready!\n")

    test_queries = [
        "How much R&D did the company spend in 2023?",
        "What was the operating income and net profit in 2024?",
        "Asia-Pacific revenue growth rate",
    ]

    for q in test_queries:
        print("=" * 75)
        print(f"Query: '{q}'")
        print("=" * 75)
        response = pipeline.retrieve(q)

        for cand in response.candidates:
            pre_rank = cand.chunk.metadata.get("pre_rerank_rank", "-")
            bm25_s = cand.scores.get("bm25", "-")
            dense_s = cand.scores.get("dense", "-")
            rrf_s = cand.scores.get("rrf", "-")
            rerank_s = cand.score
            print(
                f"[Final Rank {cand.rank}] (RRF Rank was: {pre_rank}) | "
                f"Rerank Score: {rerank_s:+.4f} | "
                f"RRF: {rrf_s} | "
                f"ID: {cand.chunk.chunk_id} | "
                f"Section: '{cand.chunk.section}'\n"
                f"    Evidence Citation: {cand.chunk.to_evidence().model_dump()}"
            )
        print()
