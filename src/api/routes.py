from fastapi import APIRouter, HTTPException

from src.schemas import RetrievalRequest, RetrievalResponse


router = APIRouter()

pipeline = None


def set_pipeline(retrieval_pipeline):
    global pipeline
    pipeline = retrieval_pipeline


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