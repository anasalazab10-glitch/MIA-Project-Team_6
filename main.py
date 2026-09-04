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