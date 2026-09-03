from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .processor import DocProcessor
from .schemas import ProcessResponse

app = FastAPI(title="doc-processor-api", version="0.1.0")

processor: DocProcessor | None = None


@app.on_event("startup")
def _startup() -> None:
    global processor
    processor = DocProcessor(lang="en")


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

    # basic extension check (not perfect, but useful)
    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id, num_pages, elements = processor.process_pdf(
        pdf_bytes=pdf_bytes,
        document_id=document_id,
        dpi=dpi,
    )
    return ProcessResponse(document_id=doc_id, num_pages=num_pages, elements=elements)
