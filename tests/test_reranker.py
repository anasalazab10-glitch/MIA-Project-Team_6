"""Unit and integration tests for Cross-Encoder Reranker and full RetrievalPipeline."""

import json
from pathlib import Path

import pytest
from src.fusion import build_hybrid_pipeline
from src.reranker import CrossEncoderReranker, RetrievalPipeline
from src.schemas import Candidate, Chunk, RetrievalMethod


@pytest.fixture
def mock_chunks():
    mock_path = Path("data/mock_chunks.json")
    with open(mock_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.model_validate(item) for item in data]


def test_cross_encoder_rerank_candidates():
    reranker = CrossEncoderReranker()

    c1 = Chunk(chunk_id="c1", document_id="doc1", page=[1], content_type="text", content="Apples and oranges are fruits.")
    c2 = Chunk(chunk_id="c2", document_id="doc1", page=[1], content_type="text", content="In 2024, Northbridge revenue reached $120 million.")

    candidates = [
        Candidate(chunk=c1, score=0.9, retrieval_method=RetrievalMethod.DENSE, rank=1),
        Candidate(chunk=c2, score=0.5, retrieval_method=RetrievalMethod.DENSE, rank=2),
    ]

    query = "What was the revenue in 2024?"
    reranked = reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    # The revenue chunk should be promoted to rank 1
    assert reranked[0].chunk.chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[0].retrieval_method == RetrievalMethod.RERANKER
    assert "reranker" in reranked[0].scores
    # Pre-rerank rank should be recorded
    assert reranked[0].chunk.metadata.get("pre_rerank_rank") == 2


def test_full_pipeline_top30_to_top5(mock_chunks):
    hybrid = build_hybrid_pipeline(mock_chunks, collection_name="test_pipeline_rerank")
    pipeline = RetrievalPipeline(hybrid_retriever=hybrid, over_retrieve_k=30, final_top_k=5)

    response = pipeline.retrieve("Operating income and net profit in 2024")

    assert response.retrieval_method == RetrievalMethod.RERANKER
    assert len(response.candidates) == 5

    # Check that each candidate produces valid evidence citations
    for cand in response.candidates:
        evidence = cand.chunk.to_evidence()
        assert evidence.document_id is not None
        assert isinstance(evidence.page, list)
