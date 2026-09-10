from fastapi import APIRouter, HTTPException
from src.indexing import index_elements
from langfuse import Langfuse
from src.schemas import (
    IndexRequest,
    IndexResponse,
    RetrievalRequest,
    RetrievalResponse,
)

langfuse = Langfuse()
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


@router.post("/search", response_model=RetrievalResponse)
def search(request: RetrievalRequest):
    if pipeline is None:
        raise HTTPException(...)

    trace = langfuse.trace(
        id=request.trace_id,
        name="retrieval-search",
        input={
            "query": request.query,
            "top_k": request.top_k,
            "retrieval_method": request.retrieval_method.value,
            "metadata_filter": request.metadata_filter,
        },
    )

    span = trace.span(
        name="retrieval-pipeline",
        input={
            "query": request.query,
            "final_top_k": request.top_k,
            "metadata_filter": request.metadata_filter,
        },
    )

    try:
        result = pipeline.retrieve(
            query=request.query,
            final_top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

        span.end(
            output={
                "num_candidates": len(result.candidates),
            }
        )

        return result

    except Exception as exc:
        span.end(
            level="ERROR",
            status_message=str(exc),
        )
        raise

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

    doc_id = request.document_id or request.elements[0].get(
        "document_id",
        "doc_unknown",
    )

    for el in request.elements:
        if "document_id" not in el:
            el["document_id"] = doc_id

    try:
        # Chunking + embeddings + Qdrant
        # Returns the exact chunks that were indexed.
        chunks = index_elements(
            elements=request.elements,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    if bm25_retriever:
        existing_chunks = getattr(
            bm25_retriever,
            "chunks",
            [],
        )

        updated_chunks = existing_chunks + chunks
        bm25_retriever.index(updated_chunks)

    return IndexResponse(
        status="indexed",
        document_id=doc_id,
        num_chunks=len(chunks),
        message=(
            f"Successfully indexed {len(chunks)} chunks "
            "into vector store and BM25."
        ),
    )
