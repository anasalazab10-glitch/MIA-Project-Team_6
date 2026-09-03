from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .processor import DocProcessor, sha1_id
from .schemas import ProcessResponse

app = FastAPI(title="doc-processor-api", version="0.1.0")

processor: DocProcessor | None = None


@app.on_event("startup")
def _startup() -> None:
    global processor
    processor = DocProcessor(lang="en")


def _default_document_id(filename: str | None, pdf_bytes: bytes) -> str:
    """Default document_id = uploaded filename without extension.

    For TAT-DQA this equals the dataset's doc uid (e.g. d5d739657785641f4689be3d234e0a8f),
    which keeps citations and evaluation aligned. Falls back to sha1 of the bytes.
    """
    if filename:
        stem = os.path.splitext(os.path.basename(filename.replace("\\", "/")))[0].strip()
        if stem:
            return stem
    return sha1_id(pdf_bytes)


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.post("/process", response_model=ProcessResponse)
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    dpi: int = Form(default=200),
) -> ProcessResponse:
    if processor is None:
        raise HTTPException(status_code=500, detail="Processor not initialized")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = (document_id or "").strip() or _default_document_id(file.filename, pdf_bytes)

    doc_id, num_pages, elements = processor.process_pdf(
        pdf_bytes=pdf_bytes,
        document_id=doc_id,
        dpi=dpi,
    )
    return ProcessResponse(document_id=doc_id, num_pages=num_pages, elements=elements)
