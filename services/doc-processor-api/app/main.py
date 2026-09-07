from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .mineru_processor import MinerUDocProcessor
from .processor import DocProcessor, sha1_id
from .schemas import ProcessResponse

app = FastAPI(title="doc-processor-api", version="0.1.0")

# Engine selection: "mineru" (default for fast inference) or "paddle" (legacy PP-Structure)
DEFAULT_ENGINE = os.getenv("DOC_PROCESSOR_ENGINE", "mineru").lower().strip()
mineru_processor: MinerUDocProcessor | None = None
paddle_processor: DocProcessor | None = None


@app.on_event("startup")
def _startup() -> None:
    global mineru_processor, paddle_processor
    mineru_processor = MinerUDocProcessor()
    if DEFAULT_ENGINE == "paddle" or os.getenv("PRELOAD_PADDLE", "false").lower() == "true":
        try:
            paddle_processor = DocProcessor(lang="en")
        except Exception:
            pass


def _default_document_id(filename: str | None, pdf_bytes: bytes) -> str:
    if filename:
        stem = os.path.splitext(os.path.basename(filename.replace("\\", "/")))[0].strip()
        if stem:
            return stem
    return sha1_id(pdf_bytes)


@app.get("/health")
def health():
    return JSONResponse({
        "status": "ok",
        "default_engine": DEFAULT_ENGINE,
        "available_engines": ["mineru", "paddle"],
    })


@app.post("/process", response_model=ProcessResponse)
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    dpi: int = Form(default=200),
    include_full_page_ocr: bool = Form(default=True),
    engine: str | None = Form(default=None),
    parallel: bool = Form(default=False),
) -> ProcessResponse:
    global mineru_processor, paddle_processor

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = (document_id or "").strip() or _default_document_id(file.filename, pdf_bytes)
    chosen_engine = (engine or DEFAULT_ENGINE).lower().strip()

    if chosen_engine == "mineru":
        if mineru_processor is None:
            mineru_processor = MinerUDocProcessor()
        doc_id, num_pages, elements = mineru_processor.process_pdf(
            pdf_bytes=pdf_bytes,
            document_id=doc_id,
            parallel=parallel,
        )
    else:
        if paddle_processor is None:
            paddle_processor = DocProcessor(lang="en")
        doc_id, num_pages, elements = paddle_processor.process_pdf(
            pdf_bytes=pdf_bytes,
            document_id=doc_id,
            dpi=dpi,
            include_full_page_ocr=include_full_page_ocr,
        )

    return ProcessResponse(document_id=doc_id, num_pages=num_pages, elements=elements)

