"""
main.py

Exposes the agent as a microservice: POST /run accepts a question, runs it
through the compiled LangGraph (classifier -> retriever -> reasoner ->
formatter, with retry/insufficient-evidence branching), and returns the
validated JSON answer.

Test it with:
    uvicorn main:app --reload
Then in another terminal:
    curl -X POST http://localhost:8000/run -H "Content-Type: application/json" -d '{"question": "What was Jabil Circuit'\''s operating income in 2019?"}'
"""
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()  # must run BEFORE importing graph, since classifier.py creates
                # its Groq client at import time using GROQ_API_KEY

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import agent_graph

app = FastAPI(title="Reasoner Service")


class RunRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class RunResponse(BaseModel):
    answer_type: str
    evidence: list
    params: Dict[str, Any]


def run_agent_graph(question: str, session_id: Optional[str]) -> Dict[str, Any]:
    """
    Runs the real compiled LangGraph pipeline end to end:
    classifier -> retriever -> (reasoner | mark_insufficient) -> formatter.
    """
    initial_state = {
        "question": question,
        "session_id": session_id,
        "retrieved_chunks": [],
    }
    result_state = agent_graph.invoke(initial_state)
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