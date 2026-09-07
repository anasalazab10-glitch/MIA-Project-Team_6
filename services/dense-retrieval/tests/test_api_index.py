"""Unit tests for the POST /index endpoint in retrieval-api."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api.routes import router, set_pipeline
from fastapi import FastAPI


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


def test_index_endpoint_success(app):
    mock_pipeline = MagicMock()
    mock_emb_model = MagicMock()
    mock_emb_model.encode_chunks.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_bm25 = MagicMock()
    mock_bm25.chunks = []

    set_pipeline(
        retrieval_pipeline=mock_pipeline,
        emb_model=mock_emb_model,
        vec_store=mock_vector_store,
        bm25=mock_bm25,
    )

    client = TestClient(app)

    payload = {
        "document_id": "test_doc",
        "elements": [
            {
                "chunk_id": "el1",
                "document_id": "test_doc",
                "page": [1],
                "content_type": "heading",
                "content": "Financial Highlights",
            },
            {
                "chunk_id": "el2",
                "document_id": "test_doc",
                "page": [1],
                "content_type": "text",
                "content": "Revenue grew by 25% year-over-year.",
            },
            {
                "chunk_id": "el3",
                "document_id": "test_doc",
                "page": [2],
                "content_type": "table",
                "section": "Balance Sheet",
                "content": {
                    "headers": ["Item", "2024"],
                    "rows": [["Assets", "1000"]],
                },
            },
        ],
    }

    res = client.post("/index", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "indexed"
    assert data["document_id"] == "test_doc"
    assert data["num_chunks"] == 2  # 1 text chunk + 1 table chunk

    # Verify vector store and BM25 index were called
    assert mock_vector_store.add_chunks.called
    assert mock_bm25.index.called


def test_index_endpoint_empty_elements(app):
    client = TestClient(app)
    res = client.post("/index", json={"document_id": "doc1", "elements": []})
    assert res.status_code == 400
