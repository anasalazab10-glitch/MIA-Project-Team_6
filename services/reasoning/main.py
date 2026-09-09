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
from langfuse import  Langfuse
from dotenv import load_dotenv
load_dotenv()  # must run BEFORE importing graph, since classifier.py creates
                # its Groq client at import time using GROQ_API_KEY

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import agent_graph

langfuse = Langfuse()
app = FastAPI(title="Reasoner Service")


class RunRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    trace_id: Optional[str] = None


class RunResponse(BaseModel):
    answer_type: str
    evidence: list
    params: Dict[str, Any]
    retrieved_candidates: List[Dict[str, Any]] = []


def run_agent_graph(
    question: str,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the real compiled LangGraph pipeline end to end:
    classifier -> retriever -> (reasoner | mark_insufficient) -> formatter.
    """
    initial_state = {
        "question": question,
        "session_id": session_id,
        "document_id": document_id,
        "trace_id": trace_id,
        "retrieved_chunks": [],
    }
    trace = langfuse.trace(
            id=trace_id,
            name="reasoning-run",
            input={
                "question": question,
                "session_id": session_id,
                "document_id": document_id,
            },
        )
        
    span = trace.span(
            name="reasoning-agent-graph",
            input=initial_state,
        )
        
    try:
            result_state = agent_graph.invoke(initial_state)
        
            span.end(
                output={
                    "status": "completed",
                }
            )
    except Exception as exc:
            span.end(
                level="ERROR",
                status_message=str(exc),
            )
            raise
    final_answer = result_state.get("final_answer", {})
    if not isinstance(final_answer, dict):
        final_answer = final_answer.model_dump() if hasattr(final_answer, "model_dump") else {}

    # Extract retrieved candidates reflecting exactly what the reasoning agent used
    retrieved_chunks = result_state.get("retrieved_chunks", [])
    candidates = []
    seen = set()
    for c in retrieved_chunks:
        c_dict = c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else dict(c))
        # Ensure chunk object is also present for schema compatibility
        if "chunk" not in c_dict:
            c_dict["chunk"] = {
                "chunk_id": c_dict.get("chunk_id"),
                "document_id": c_dict.get("document_id"),
                "page": c_dict.get("page"),
                "section": c_dict.get("section"),
                "content_type": c_dict.get("content_type"),
                "text": c_dict.get("text"),
            }
        dedup_key = c_dict.get("chunk_id") or (
            c_dict.get("document_id"),
            str(c_dict.get("page")),
            c_dict.get("text", "")[:40],
        )
        if dedup_key not in seen:
            seen.add(dedup_key)
            candidates.append(c_dict)

    final_answer["retrieved_candidates"] = candidates
    return final_answer


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)

def run(request: RunRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = run_agent_graph(
            question=request.question,
            session_id=request.session_id,
            document_id=request.document_id,
            trace_id=request.trace_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")

    return answer