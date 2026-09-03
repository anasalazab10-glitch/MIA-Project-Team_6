# doc-processor-api

FastAPI microservice that converts raw PDF bytes into structured elements for RAG:
- text blocks
- headings
- tables (structured: headers + rows)
- page numbers (1-based)
- normalized bounding boxes for citation/highlighting

## Endpoints

- GET /health
- POST /process (multipart/form-data)
  - file: PDF file
  - document_id (optional)
  - dpi (optional, default=200)

## Run (WSL)

Recommended: create python 3.10 env (e.g. micromamba) and install requirements.

Example:

micromamba run -n ledger-docproc python -m pip install -r services/doc-processor-api/requirements.txt

micromamba run -n ledger-docproc uvicorn services.doc-processor-api.app.main:app --host 0.0.0.0 --port 8001
