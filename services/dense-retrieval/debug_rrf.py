from src.vector_store import VectorStore
from src.embeddings import EmbeddingModel
from src.dense_retriever import DenseRetriever
from src.bm25_retriever import BM25Retriever
from src.fusion import reciprocal_rank_fusion, get_query_weights


QUERY = (
    "What was the absolute difference between Jabil's gross profit and Advanced Energy's in 2018?"
)


def print_chunk_content(candidate, source, rank):
    chunk = candidate.chunk

    print()
    print("-" * 80)
    print(f"{source} RANK: {rank}")
    print(f"Score: {candidate.score}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Document ID: {chunk.document_id}")
    print(f"Page: {chunk.page}")
    print(f"Section: {chunk.section}")
    print(f"Content Type: {chunk.content_type}")
    print("-" * 80)

    print("CONTENT:")
    
    if isinstance(chunk.content, str):
        print(chunk.content)
    else:
        print(chunk.content)

    print("-" * 80)


def main():
    print("=" * 80)
    print("RRF RETRIEVAL DEBUG")
    print("=" * 80)
    print(f"QUERY: {QUERY}")
    print()

    # ------------------------------------------------------------------
    # Load real chunks from Qdrant Cloud
    # ------------------------------------------------------------------
    vector_store = VectorStore()
    chunks = vector_store.get_all_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------
    bm25 = BM25Retriever(chunks)

    bm25_results = bm25.search(
        query=QUERY,
        top_k=30,
    )

    print()
    print("=" * 80)
    print("BM25 TOP 30")
    print("=" * 80)

    for c in bm25_results:
        print_chunk_content(
            candidate=c,
            source="BM25",
            rank=c.rank,
        )

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Loading embedding model...")
    print("=" * 80)

    embedding_model = EmbeddingModel()

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    dense_results = dense.retrieve(
        query=QUERY,
        top_k=30,
    ).candidates

    print()
    print("=" * 80)
    print("DENSE TOP 30")
    print("=" * 80)

    for c in dense_results:
        print_chunk_content(
            candidate=c,
            source="DENSE",
            rank=c.rank,
        )

    # ------------------------------------------------------------------
    # Query-aware weights
    # ------------------------------------------------------------------
    query_weights = get_query_weights(QUERY)

    print()
    print("=" * 80)
    print("QUERY-AWARE RRF WEIGHTS")
    print("=" * 80)

    print(f"BM25 weight : {query_weights['bm25']}")
    print(f"Dense weight: {query_weights['dense']}")

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------
    fused = reciprocal_rank_fusion(
        ranked_lists={
            "bm25": bm25_results,
            "dense": dense_results,
        },
        k=60,
        weights=query_weights,
        top_k=30,
    )

    print()
    print("=" * 80)
    print("RRF TOP 30")
    print("=" * 80)

    for c in fused:
        b_rank = c.chunk.metadata.get("bm25_rank", "-")
        d_rank = c.chunk.metadata.get("dense_rank", "-")

        print()
        print("-" * 80)
        print(f"RRF RANK: {c.rank}")
        print(f"RRF SCORE: {c.score:.6f}")
        print(f"BM25 RANK: {b_rank}")
        print(f"DENSE RANK: {d_rank}")
        print(f"Chunk ID: {c.chunk.chunk_id}")
        print(f"Document ID: {c.chunk.document_id}")
        print(f"Page: {c.chunk.page}")
        print(f"Section: {c.chunk.section}")
        print(f"Content Type: {c.chunk.content_type}")
        print("-" * 80)

        print("CONTENT:")
        
        if isinstance(c.chunk.content, str):
            print(c.chunk.content)
        else:
            print(c.chunk.content)

        print("-" * 80)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()