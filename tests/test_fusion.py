"""Unit and integration tests for Reciprocal Rank Fusion (RRF) and HybridRetriever."""

import json
from pathlib import Path

import pytest
from src.bm25_retriever import BM25Retriever
from src.dense_retrieval import DenseRetriever
from src.embeddings import EmbeddingModel
from src.fusion import HybridRetriever, RRFFusion, reciprocal_rank_fusion
from src.schemas import Candidate, Chunk, RetrievalMethod
from src.vector_store import VectorStore


@pytest.fixture
def mock_chunks():
    mock_path = Path("data/mock_chunks.json")
    with open(mock_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.model_validate(item) for item in data]


def test_rrf_math_calculation():
    # Build two mock candidates
    c1 = Chunk(chunk_id="chunk_a", document_id="doc1", page=[1], content_type="text", content="A")
    c2 = Chunk(chunk_id="chunk_b", document_id="doc1", page=[1], content_type="text", content="B")

    # List 1: chunk_a (rank 1), chunk_b (rank 2)
    list1 = [
        Candidate(chunk=c1, score=10.0, retrieval_method=RetrievalMethod.BM25, rank=1),
        Candidate(chunk=c2, score=5.0, retrieval_method=RetrievalMethod.BM25, rank=2),
    ]

    # List 2: chunk_b (rank 1), chunk_a (rank 2)
    list2 = [
        Candidate(chunk=c2, score=0.9, retrieval_method=RetrievalMethod.DENSE, rank=1),
        Candidate(chunk=c1, score=0.8, retrieval_method=RetrievalMethod.DENSE, rank=2),
    ]

    # RRF with k=60:
    # chunk_a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # chunk_b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.032522 (tie)
    fused = reciprocal_rank_fusion({"bm25": list1, "dense": list2}, k=60)

    assert len(fused) == 2
    assert fused[0].retrieval_method == RetrievalMethod.RRF
    assert fused[0].score == pytest.approx(0.032522, rel=1e-3)
    assert "bm25" in fused[0].scores
    assert "dense" in fused[0].scores
    assert "rrf" in fused[0].scores


def test_rrf_preserves_intermediate_ranks_and_metadata():
    c1 = Chunk(chunk_id="c1", document_id="d1", page=[1], content_type="text", content="Text 1")
    cand1 = Candidate(chunk=c1, score=8.5, retrieval_method=RetrievalMethod.BM25, rank=1)
    cand2 = Candidate(chunk=c1, score=0.88, retrieval_method=RetrievalMethod.DENSE, rank=3)

    fused = reciprocal_rank_fusion({"bm25": [cand1], "dense": [cand2]}, k=60)
    assert len(fused) == 1
    assert fused[0].chunk.metadata.get("bm25_rank") == 1
    assert fused[0].chunk.metadata.get("dense_rank") == 3


def test_hybrid_retriever_pipeline(mock_chunks):
    # 1. BM25
    bm25 = BM25Retriever(mock_chunks)

    # 2. Dense
    embed_model = EmbeddingModel()
    embeddings = embed_model.encode_chunks(mock_chunks)
    vector_store = VectorStore(collection_name="test_fusion_hybrid", vector_size=384)
    vector_store.add_chunks(mock_chunks, embeddings)
    dense = DenseRetriever(embed_model, vector_store)

    # 3. Hybrid
    hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    response = hybrid.search("Asia-Pacific revenue growth rate", top_k=5)

    assert response.retrieval_method == RetrievalMethod.RRF
    assert len(response.candidates) == 5
    assert response.candidates[0].rank == 1
    assert response.candidates[0].score > 0
