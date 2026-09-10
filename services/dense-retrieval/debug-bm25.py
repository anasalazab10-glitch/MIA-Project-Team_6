from src.vector_store import VectorStore
from src.bm25_retriever import BM25Retriever


QUERY = "What was A10 Networks' total revenue between 2015 to 2019?"


def main():
    print("=" * 80)
    print("BM25 RETRIEVAL DEBUG")
    print("=" * 80)
    print(f"QUERY: {QUERY}")
    print()

    print("Connecting to Qdrant...")
    vector_store = VectorStore()

    print("Loading chunks from Qdrant...")
    chunks = vector_store.get_all_chunks()

    print(f"Loaded {len(chunks)} chunks.")
    print()

    print("Building BM25 index...")
    retriever = BM25Retriever(chunks)

    print(f"BM25 corpus size: {retriever.corpus_size}")
    print(f"Average document length: {retriever.avg_doc_len:.2f}")
    print()

    print("Query tokens:")
    print(retriever.search.__self__ if False else "")
    
    from src.bm25_retriever import financial_tokenizer
    tokens = financial_tokenizer(QUERY)
    print(tokens)
    print()

    print("Searching BM25...")
    results = retriever.search(
        query=QUERY,
        top_k=30,
    )

    print()
    print("=" * 80)
    print(f"RETRIEVED {len(results)} BM25 CANDIDATES")
    print("=" * 80)

    for candidate in results:
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
