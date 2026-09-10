from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.dense_retriever import DenseRetriever


QUERY = "WWhat were Teekay Corporation's Realized losses in 2019, 2018 and 2017 respectively?"


def main():
    print("=" * 80)
    print("DENSE RETRIEVAL DEBUG")
    print("=" * 80)
    print(f"QUERY: {QUERY}")
    print()

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    print("Connecting to Qdrant...")
    vector_store = VectorStore()

    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    print("Searching Qdrant...")
    response = retriever.retrieve(
        query=QUERY,
        top_k=30,
    )

    print()
    print("=" * 80)
    print(f"RETRIEVED {len(response.candidates)} DENSE CANDIDATES")
    print("=" * 80)

    for candidate in response.candidates:
        chunk = candidate.chunk

        print()
        print("-" * 80)
        print(f"RANK: {candidate.rank}")
        print(f"SCORE: {candidate.score:.6f}")
        print(f"DOCUMENT: {chunk.document_id}")
        print(f"CHUNK: {chunk.chunk_id}")
        print(f"PAGE: {chunk.page}")
        print(f"SECTION: {chunk.section}")
        print(f"TYPE: {chunk.content_type}")
        print()
        print("CONTENT:")
        print(chunk.to_text())


if __name__ == "__main__":
    main()
