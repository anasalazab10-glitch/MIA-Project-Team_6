from fastapi import APIRouter, HTTPException

from src.chunking import create_chunks
from src.schemas import (
    IndexRequest,
    IndexResponse,
    RetrievalRequest,
    RetrievalResponse,
)


router = APIRouter()

pipeline = None
embedding_model = None
vector_store = None
bm25_retriever = None


def set_pipeline(
    retrieval_pipeline,
    emb_model=None,
    vec_store=None,
    bm25=None,
):
    global pipeline, embedding_model, vector_store, bm25_retriever
    pipeline = retrieval_pipeline
    embedding_model = emb_model
    vector_store = vec_store
    bm25_retriever = bm25


@router.get("/health")
def health():
    return {
        "status": "ok"
    }


@router.post(
    "/search",
    response_model=RetrievalResponse,
)
def search(request: RetrievalRequest):

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval pipeline is not initialized.",
        )

    return pipeline.retrieve(
        query=request.query,
        final_top_k=5,
        metadata_filter=request.metadata_filter,
    )


@router.post(
    "/index",
    response_model=IndexResponse,
)
def index(request: IndexRequest):
    """
    Dynamically ingest and index document elements:
    1. Converts processed elements into standardized retrieval chunks.
    2. Generates embeddings and stores chunks + vectors in Qdrant.
    3. Updates the live BM25 inverted index.
    """
    if not request.elements:
        raise HTTPException(
            status_code=400,
            detail="Elements list cannot be empty.",
        )

    doc_id = request.document_id or request.elements[0].get("document_id", "doc_unknown")
    for el in request.elements:
        if "document_id" not in el:
            el["document_id"] = doc_id

    chunks = create_chunks(request.elements)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks could be generated from the elements.",
        )

    if embedding_model and vector_store:
        embeddings = embedding_model.encode_chunks(chunks)
        vector_store.add_chunks(chunks, embeddings)

    if bm25_retriever:
        existing_chunks = getattr(bm25_retriever, "chunks", [])
        updated_chunks = existing_chunks + chunks
        bm25_retriever.index(updated_chunks)

    return IndexResponse(
        status="indexed",
        document_id=doc_id,
        num_chunks=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks into vector store and BM25.",
    )