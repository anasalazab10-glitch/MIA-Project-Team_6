"""
main.py

Exposes the agent as a microservice: POST /run accepts a question, runs it
through the compiled LangGraph, formats it, and returns the validated JSON
answer.

STATUS: graph.py isn't built yet, so this calls the same placeholder logic
as run_test.py. Once graph.py exists, replace `run_agent_graph()` below.

Test it right now with:
    uvicorn main:app --reload
Then in another terminal:
    curl -X POST http://localhost:8000/run -H "Content-Type: application/json" -d '{"question": "What was revenue in 2020?"}'
"""
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from formatter import format_answer

app = FastAPI(title="Reasoner Service")


class RunRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class RunResponse(BaseModel):
    answer_type: str
    evidence: list
    params: Dict[str, Any]


# PLACEHOLDER
def run_agent_graph(question: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    TEMPORARY stand-in for the real compiled LangGraph. Returns a fake
    'direct' answer so the rest of the harness can be built/tested now.
    """
    fake_state = {
        "question": question,
        "session_id": session_id,
        "question_type": "direct",
        "extracted_values": ["MOCK_ANSWER"],
        "evidence": [{"document_id": "mock_doc", "page": 1, "section": "General"}],
    }
    result_state = format_answer(fake_state)
    return result_state["final_answer"]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = run_agent_graph(request.question, request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")

    return answer
