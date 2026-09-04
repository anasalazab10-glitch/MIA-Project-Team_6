from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.bm25_retriever import BM25Retriever
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.dense_retriever import DenseRetriever
from src.fusion import HybridRetriever, RRFFusion
from src.reranker import CrossEncoderReranker, RetrievalPipeline

from src.api.routes import router, set_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Ledger Retrieval API...")

    # 1. Connect to existing Qdrant
    vector_store = VectorStore(
        collection_name="ledger_chunks",
        vector_size=384,
        host="localhost",
        port=6333,
    )

    print("Connected to Qdrant.")


    # 2. Load existing chunks from Qdrant
    chunks = vector_store.get_all_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks found in Qdrant. "
            "Run the indexing pipeline first."
        )

    print(f"Loaded {len(chunks)} chunks from Qdrant.")


    # 3. Build BM25 index from existing chunks

    bm25_retriever = BM25Retriever(chunks)

    print("BM25 index built.")

    # --------------------------------------------------
    # 4. Load embedding model
    # --------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("Embedding model loaded.")

    # --------------------------------------------------
    # 5. Create Dense Retriever
    # --------------------------------------------------

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    print("Dense retriever ready.")

    # --------------------------------------------------
    # 6. Create RRF Fusion
    # --------------------------------------------------

    fusion = RRFFusion(
        k=60,
        weights={
            "bm25": 1.0,
            "dense": 1.0,
        },
        default_top_k=30,
    )

    # --------------------------------------------------
    # 7. Create Hybrid Retriever
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        fusion=fusion,
    )

    print("Hybrid retriever ready.")

    # --------------------------------------------------
    # 8. Create Cross-Encoder Reranker
    # --------------------------------------------------

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("Reranker ready.")

    # --------------------------------------------------
    # 9. Create final retrieval pipeline
    # --------------------------------------------------

    retrieval_pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        over_retrieve_k=30,
        final_top_k=5,
    )

    # Give pipeline to routes
    set_pipeline(retrieval_pipeline)

    print("Full retrieval pipeline is ready.")

    yield

    print("Shutting down Ledger Retrieval API...")


app = FastAPI(
    title="Ledger Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)