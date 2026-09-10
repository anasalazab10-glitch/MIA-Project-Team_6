from src.vector_store import VectorStore
from src.embeddings import EmbeddingModel
from src.dense_retriever import DenseRetriever
from src.bm25_retriever import BM25Retriever
from src.fusion import reciprocal_rank_fusion, get_query_weights
from src.reranker import CrossEncoderReranker


QUERY = (
    "What were Lam Research's reasons for the net loss on extinguishment of debt realized in the year ended June 25, 2017? Which page supports the answer?"
)


def get_final_top5():
    # Load real chunks
    vector_store = VectorStore()
    chunks = vector_store.get_all_chunks()

    # BM25 top 30
    bm25 = BM25Retriever(chunks)
    bm25_results = bm25.search(
        query=QUERY,
        top_k=30,
    )

    # Dense top 30
    embedding_model = EmbeddingModel()

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    dense_results = dense.retrieve(
        query=QUERY,
        top_k=30,
    ).candidates

    # RRF top 30
    query_weights = get_query_weights(QUERY)

    fused = reciprocal_rank_fusion(
        ranked_lists={
            "bm25": bm25_results,
            "dense": dense_results,
        },
        k=60,
        weights=query_weights,
        top_k=30,
    )

    # Real CrossEncoder reranker
    reranker = CrossEncoderReranker()

    # CrossEncoder: 30 candidates -> final 5
    top_5 = reranker.rerank(
        query=QUERY,
        candidates=fused,
        top_k=5,
    )

    return top_5


if __name__ == "__main__":
    top_5 = get_final_top5()

    for rank, candidate in enumerate(top_5, start=1):
        chunk = candidate.chunk

        print("=" * 80)
        print(f"FINAL RANK: {rank}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Page: {chunk.page}")
        print(f"Section: {chunk.section}")
        print(f"Content Type: {chunk.content_type}")
        print(f"CrossEncoder Score: {candidate.score}")
        print("-" * 80)
        print("CONTENT:")
        print(chunk.to_text())
        print("=" * 80)
