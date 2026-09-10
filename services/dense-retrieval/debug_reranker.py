from src.vector_store import VectorStore
from src.embeddings import EmbeddingModel
from src.dense_retriever import DenseRetriever
from src.bm25_retriever import BM25Retriever
from src.fusion import reciprocal_rank_fusion
from src.reranker import CrossEncoderReranker

QUERY = "For Microsoft, how much were the top 3 components of property and equipment as a % of the total at cost, property and equipment for 2019?"

TARGET = "d82825dc611851d39f74ecf5a5749e32_chunk8"


def main():
    print("=" * 80)
    print("CROSS-ENCODER RERANKER DEBUG")
    print("=" * 80)
    print(f"QUERY: {QUERY}")
    print(f"TARGET: {TARGET}")
    print()

    # ---------------------------------------------------------
    # 1. Load chunks
    # ---------------------------------------------------------
    vector_store = VectorStore()
    chunks = vector_store.get_all_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    # ---------------------------------------------------------
    # 2. BM25
    # ---------------------------------------------------------
    bm25 = BM25Retriever(chunks)

    bm25_results = bm25.search(
        query=QUERY,
        top_k=30,
    )

    # ---------------------------------------------------------
    # 3. Dense
    # ---------------------------------------------------------
    print()
    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    dense_results = dense.retrieve(
        query=QUERY,
        top_k=30,
    ).candidates

    # ---------------------------------------------------------
    # 4. RRF
    # ---------------------------------------------------------
    fused = reciprocal_rank_fusion(
        ranked_lists={
            "bm25": bm25_results,
            "dense": dense_results,
        },
        k=60,
        weights={
            "bm25": 1.0,
            "dense": 1.0,
        },
        top_k=30,
    )

    print()
    print("=" * 80)
    print("RRF TARGET")
    print("=" * 80)

    for candidate in fused:
        if candidate.chunk.chunk_id == TARGET:
            print(f"RRF rank : {candidate.rank}")
            print(f"RRF score: {candidate.score:.6f}")
            print(f"Content  :")
            print(candidate.chunk.to_text())

    # ---------------------------------------------------------
    # 5. CrossEncoder
    # ---------------------------------------------------------
    print()
    print("=" * 80)
    print("Loading CrossEncoder...")
    print("=" * 80)

    reranker = CrossEncoderReranker()

    # IMPORTANT:
    # We manually score all 30 RRF candidates so we can
    # inspect where the target goes.
    pairs = [
        [
            QUERY,
            reranker._prepare_document_text(candidate.chunk)
        ]
        for candidate in fused
    ]

    raw_scores = reranker.model.predict(pairs)

    if hasattr(raw_scores, "tolist"):
        scores = raw_scores.tolist()
    else:
        scores = list(raw_scores)

    scored = []

    for candidate, score in zip(fused, scores):
        scored.append(
            (
                candidate,
                float(score),
            )
        )

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # ---------------------------------------------------------
    # 6. Print complete reranker ranking
    # ---------------------------------------------------------
    print()
    print("=" * 80)
    print("CROSS-ENCODER RANKING")
    print("=" * 80)

    target_found = False

    for rank, (candidate, score) in enumerate(scored, start=1):

        chunk_id = candidate.chunk.chunk_id
        rrf_rank = candidate.rank
        rrf_score = candidate.score

        marker = ""

        if chunk_id == TARGET:
            marker = "  <<< TARGET"
            target_found = True

        print(
            f"FINAL/RERANK RANK={rank:2d} | "
            f"RERANK SCORE={score:+.6f} | "
            f"RRF RANK={rrf_rank:2d} | "
            f"RRF SCORE={rrf_score:.6f} | "
            f"CHUNK={chunk_id}"
            f"{marker}"
        )

    # ---------------------------------------------------------
    # 7. Target details
    # ---------------------------------------------------------
    print()
    print("=" * 80)

    for rank, (candidate, score) in enumerate(scored, start=1):

        if candidate.chunk.chunk_id == TARGET:

            print("TARGET ANALYSIS")
            print("=" * 80)

            print(f"RRF rank:        {candidate.rank}")
            print(f"RRF score:       {candidate.score:.6f}")
            print(f"Reranker rank:   {rank}")
            print(f"Reranker score:  {score:+.6f}")

            print()
            print("TARGET CONTENT:")
            print(candidate.chunk.to_text())

            break

    if not target_found:
        print("TARGET DISAPPEARED BEFORE RERANKING!")

    print("=" * 80)


if __name__ == "__main__":
    main()
