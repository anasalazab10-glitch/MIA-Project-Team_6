"""
Validator Service / main.py

FastAPI entry point for the Answer Validator Service. This service exposes a single validation endpoint:

  POST /validate_answer
    Accepts a JSON payload containing answer_type, evidence, and
    params. Delegates all schema logic to validator.py, then logs
    the outcome to the console in the format:

      [ANSWER-VALIDATOR-SUCCESS] Received and validated answer of
          type '<type>' with evidence { document_id, page }
      [ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: <reason>

    Always returns HTTP 200 with a JSON body { valid, reason } —
    it never raises 4xx/5xx for a schema violation, because a bad
    answer is an expected pipeline event, not a server error.

  GET /health
    Lightweight liveness check for the orchestrator.

Run with:
  uvicorn main:app --reload --port 8002

<Port number can be changed later>
"""

from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from validator import validate_answer


app = FastAPI(
    title="Answer Validator Service",
    version="0.1.0",
)


class ValidationRequest(BaseModel):
    answer_type: str
    evidence: list
    params: Dict[str, Any]


class ValidationResponse(BaseModel):
    valid: bool
    reason: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate_answer", response_model=ValidationResponse)
def validate(request: ValidationRequest):
    payload = request.model_dump()

    valid, reason = validate_answer(payload)

    if valid:
        evidence = payload["evidence"]

        if evidence:
            citation = evidence[0]
            print(
                f"[ANSWER-VALIDATOR-SUCCESS] "
                f"Received and validated answer of type "
                f"'{payload['answer_type']}' with evidence "
                f"{{ document_id: {citation.get('document_id')}, "
                f"page: {citation.get('page')} }}"
            )
        else:
            print(
                f"[ANSWER-VALIDATOR-SUCCESS] "
                f"Received and validated answer of type "
                f"'{payload['answer_type']}' with no evidence required"
            )

    else:
        print(
            f"[ANSWER-VALIDATOR-ERROR] "
            f"Invalid answer. Reason: {reason}"
        )

    return {
        "valid": valid,
        "reason": reason,
    }

