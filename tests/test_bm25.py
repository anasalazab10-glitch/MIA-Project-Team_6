"""Unit and integration tests for BM25 Keyword Retrieval."""

import json
from pathlib import Path

import pytest
from src.bm25_retriever import BM25Retriever, financial_tokenizer
from src.schemas import Chunk, ContentType, RetrievalMethod


@pytest.fixture
def mock_chunks():
    mock_path = Path("data/mock_chunks.json")
    with open(mock_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.model_validate(item) for item in data]


def test_financial_tokenizer_preserves_currencies_and_percentages():
    text = "Revenue grew 20% to $120M in 2024 from -$5M in Q1."
    tokens = financial_tokenizer(text)
    
    assert "$120m" in tokens
    assert "120m" in tokens
    assert "120" in tokens
    assert "20%" in tokens
    assert "2024" in tokens
    assert "-$5m" in tokens or "$5m" in tokens


def test_financial_tokenizer_acronym_expansion():
    # R&D expansion
    tokens_query = financial_tokenizer("How much R&D was spent?")
    assert "research" in tokens_query
    assert "development" in tokens_query
    assert "r&d" in tokens_query

    # Bidirectional expansion from document text
    tokens_doc = financial_tokenizer("Research and Development expenses were high.")
    assert "r&d" in tokens_doc or "rd" in tokens_doc


def test_bm25_search_ranking(mock_chunks):
    retriever = BM25Retriever(mock_chunks)
    results = retriever.search("operating income in 2024", top_k=3)

    assert len(results) > 0
    assert results[0].retrieval_method == RetrievalMethod.BM25
    assert results[0].rank == 1
    assert results[0].score > 0
    assert "bm25" in results[0].scores

    # Top hit should be from Financial Results
    assert results[0].chunk.section == "Financial Results"


def test_bm25_search_tables_filter(mock_chunks):
    retriever = BM25Retriever(mock_chunks)
    results = retriever.search_tables("operating expenses", top_k=2)

    assert len(results) > 0
    for cand in results:
        assert cand.chunk.content_type == ContentType.TABLE


def test_bm25_metadata_filter(mock_chunks):
    retriever = BM25Retriever(mock_chunks)
    results = retriever.search("revenue", metadata_filter={"document_id": "doc2"}, top_k=5)

    assert len(results) > 0
    for cand in results:
        assert cand.chunk.document_id == "doc2"


def test_bm25_empty_query_or_corpus(mock_chunks):
    retriever = BM25Retriever([])
    assert retriever.search("any query") == []

    non_empty = BM25Retriever(mock_chunks)
    assert non_empty.search("") == []
