# LEDGER — UI Service

Gradio-based user interface for the LEDGER financial document intelligence agent.

## Current Status

**Chat-only implementation.** The Documents and Dashboard views described in the
project spec are not yet built — this will be extended once the orchestrator
service is live and a real document-listing endpoint exists.

## What it does

- Provides a chat interface where a user can ask a question about the indexed
  financial reports.
- Sends the question to the orchestrator service via `POST /run`.
- Displays the returned answer with:
  - A colored badge showing the answer type (Direct / Calculated / Multi-item /
    Insufficient Evidence)
  - The answer value (and formula, for calculated answers)
  - The cited source document(s) and page(s)

## Placeholder / Mock Behavior

`orchestrator/main.py` does not exist yet. Until it does, `call_backend()`
in `app.py` will fail to connect and fall back to a mock
`insufficient_evidence` response, so the UI can be built and tested
independently. No code changes will be needed once the orchestrator is live —
`call_backend()` already tries the real endpoint first.

## Running Locally

```bash
cd services/UI
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` in your browser.

## Running via Docker

From the project root:

```bash
docker compose up ui
```

The UI will be available at `http://localhost:7860`.

## Configuration

| Environment Variable | Default                    | Purpose                                |
| --------------------- | --------------------------- | --------------------------------------- |
| `ORCHESTRATOR_URL`     | `http://localhost:8003`     | Base URL of the orchestrator service. Inside Docker, this should be set to `http://orchestrator:8000` per the project README. |

## Assumed Orchestrator Contract

Since `orchestrator/main.py` doesn't exist yet, this UI assumes the following
contract (mirroring the Reasoning service's own `/run` endpoint). **This needs
to be confirmed with whoever builds the orchestrator:**

**Request:**
```json
POST /run
{
  "question": "What was Jabil Circuit's operating income in 2019?"
}
```

**Response:**
```json
{
  "answer_type": "direct",
  "evidence": [
    {"document_id": "jabil-circuit-inc_2019.pdf", "page": [1], "section": "Financial Statements"}
  ],
  "params": {"value": "714200"}
}
```

## Not Yet Implemented (per project spec)

- [ ] Document interface — inspect indexed documents
- [ ] Dashboard — indexed document count, detected tables, recent queries with latency

## Dependencies

Pinned in `requirements.txt`:
- `gradio==6.26.0`
- `requests`
- `python-dotenv`

Note: `gradio` version is pinned deliberately. The Chatbot component's API
changed significantly between Gradio versions (message format), so an
unpinned version can break the app on a fresh install.