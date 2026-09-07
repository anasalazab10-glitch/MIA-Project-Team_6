"""
Unit and integration tests for the LEDGER Orchestrator Service.
Tests routing, validation safety, fallbacks, ingestion, and dashboard stats.
"""

from __future__ import annotations

import io
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from main import app, metadata_store, clients
from schemas import DocumentDetail, TableSummary


@pytest.fixture
def client():
    # Use TestClient with app
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    with patch.object(
        clients,
        "check_health",
        new_callable=AsyncMock,
        return_value={
            "doc-processor": {"reachable": True, "status_code": 200, "url": "http://test:8002/health"},
            "retrieval": {"reachable": True, "status_code": 200, "url": "http://test:8000/health"},
            "reasoning": {"reachable": True, "status_code": 200, "url": "http://test:8001/health"},
            "validator": {"reachable": True, "status_code": 200, "url": "http://test:8004/health"},
        },
    ):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "orchestrator-api"
        assert len(data["downstream_services"]) == 4


def test_run_valid_direct_answer(client):
    """Test successful query flow where reasoning produces a valid direct answer."""
    mock_reasoning_answer = {
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
        "params": {"value": "$142.5M"},
        "retrieved_candidates": [
            {
                "document_id": "doc_017",
                "page": [1],
                "section": "Income Statement",
                "content_type": "table",
                "text": "Operating income was $142.5M in 2020.",
                "score": 0.92,
                "rank": 1,
            }
        ],
    }
    mock_validator_response = {"valid": True, "reason": "Passed strict validation"}

    with patch.object(clients, "run_reasoning", new_callable=AsyncMock, return_value=mock_reasoning_answer), \
         patch.object(clients, "validate_answer", new_callable=AsyncMock, return_value=mock_validator_response):
        
        res = client.post("/run", json={"question": "What was the operating income in 2020?"})
        assert res.status_code == 200
        data = res.json()

        assert data["answer_type"] == "direct"
        assert data["params"]["value"] == "$142.5M"
        assert data["validation_status"] == "valid"
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["document_id"] == "doc_017"
        assert "latency_ms" in data
        assert len(data["retrieved_candidates"]) == 1
        assert data["retrieved_candidates"][0]["document_id"] == "doc_017"
        assert data["retrieved_candidates"][0]["score"] == 0.92


def test_run_valid_calculated_answer(client):
    """Test successful query flow for arithmetic calculation with formula."""
    mock_reasoning_answer = {
        "answer_type": "calculated",
        "evidence": [
            {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
            {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
        ],
        "params": {"value": 13.4, "formula": "(3875-3410)/3410*100"},
        "retrieved_candidates": [
            {
                "document_id": "doc_041",
                "page": [2],
                "section": "Operating Expenses",
                "content_type": "table",
                "text": "Expenses 2019: 3410, 2020: 3875",
                "score": 0.88,
                "rank": 1,
            }
        ],
    }
    mock_validator_response = {"valid": True, "reason": "Formula and citations valid"}

    with patch.object(clients, "run_reasoning", new_callable=AsyncMock, return_value=mock_reasoning_answer), \
         patch.object(clients, "validate_answer", new_callable=AsyncMock, return_value=mock_validator_response):
        
        res = client.post("/run", json={"question": "What was the percentage increase?"})
        assert res.status_code == 200
        data = res.json()

        assert data["answer_type"] == "calculated"
        assert data["params"]["value"] == 13.4
        assert data["params"]["formula"] == "(3875-3410)/3410*100"
        assert data["validation_status"] == "valid"
        assert len(data["retrieved_candidates"]) == 1


def test_run_validator_rejection_safety(client):
    """
    CRITICAL TEST: When validator rejects an answer (e.g. missing citations or hallucinated math),
    orchestrator MUST convert to schema-compliant insufficient_evidence and never leak unverified data,
    WHILE preserving retrieved_candidates for failure analysis.
    """
    candidate_data = [
        {
            "document_id": "doc_099",
            "page": [5],
            "section": "Outlook",
            "content_type": "text",
            "text": "Future revenue projections are unavailable.",
            "score": 0.65,
            "rank": 1,
        }
    ]
    hallucinated_answer = {
        "answer_type": "calculated",
        "evidence": [],  # Missing required citations!
        "params": {"value": 999.9, "formula": "100+899.9"},
        "retrieved_candidates": candidate_data,
    }
    mock_validator_response = {
        "valid": False,
        "reason": "Missing required evidence citation for calculated answer",
    }

    with patch.object(clients, "run_reasoning", new_callable=AsyncMock, return_value=hallucinated_answer), \
         patch.object(clients, "validate_answer", new_callable=AsyncMock, return_value=mock_validator_response):
        
        res = client.post("/run", json={"question": "Predict the future revenue"})
        assert res.status_code == 200
        data = res.json()

        assert data["answer_type"] == "insufficient_evidence"
        assert data["validation_status"] == "rejected"
        assert "rejected by validator" in data["params"]["reason"]
        assert data["evidence"] == []
        # CRITICAL: retrieved_candidates must be preserved for failure analysis!
        assert len(data["retrieved_candidates"]) == 1
        assert data["retrieved_candidates"][0]["document_id"] == "doc_099"


def test_run_reasoning_service_failure(client):
    """Test resilient fallback when reasoning service is down or times out."""
    with patch.object(clients, "run_reasoning", new_callable=AsyncMock, side_effect=RuntimeError("Connection timeout")), \
         patch.object(clients, "validate_answer", new_callable=AsyncMock, return_value={"valid": True, "reason": "ok"}):
        
        res = client.post("/run", json={"question": "Any question"})
        assert res.status_code == 200
        data = res.json()

        assert data["answer_type"] == "insufficient_evidence"
        assert "unable to answer" in data["params"]["reason"]
        assert data["retrieved_candidates"] == []


def test_ingest_document(client):
    """Test PDF ingestion workflow."""
    fake_pdf_content = b"%PDF-1.4 test content"
    mock_doc_processor_result = {
        "document_id": "test_corp_2023",
        "num_pages": 2,
        "elements": [
            {
                "chunk_id": "el_1",
                "document_id": "test_corp_2023",
                "page": 1,
                "content_type": "heading",
                "content": "Consolidated Financials",
            },
            {
                "chunk_id": "el_2",
                "document_id": "test_corp_2023",
                "page": 2,
                "content_type": "table",
                "section": "Balance Sheet",
                "content": {
                    "headers": ["Item", "2022", "2023"],
                    "rows": [["Cash", "100", "150"], ["Debt", "50", "40"]],
                },
            },
        ],
    }

    with patch.object(clients, "process_document", new_callable=AsyncMock, return_value=mock_doc_processor_result), \
         patch.object(clients, "index_elements", new_callable=AsyncMock, return_value={"status": "indexed"}):
        
        res = client.post(
            "/ingest",
            files={"file": ("test_corp_2023.pdf", io.BytesIO(fake_pdf_content), "application/pdf")},
            data={"document_id": "test_corp_2023"},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "indexed"
        assert data["document_id"] == "test_corp_2023"
        assert data["num_pages"] == 2
        assert data["num_elements"] == 2
        assert data["num_tables"] == 1


def test_list_and_get_documents(client):
    """Test document catalog retrieval."""
    res_list = client.get("/documents")
    assert res_list.status_code == 200
    docs = res_list.json()
    assert any(d["document_id"] == "test_corp_2023" for d in docs)

    res_detail = client.get("/documents/test_corp_2023")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["document_id"] == "test_corp_2023"
    assert len(detail["tables"]) == 1
    assert detail["tables"][0]["headers"] == ["Item", "2022", "2023"]


def test_dashboard_stats(client):
    """Test corpus dashboard statistics endpoint for Gradio UI."""
    res = client.get("/dashboard/stats")
    assert res.status_code == 200
    stats = res.json()

    assert "total_documents" in stats
    assert "total_tables" in stats
    assert "total_elements" in stats
    assert "recent_queries" in stats
    assert len(stats["recent_queries"]) > 0
