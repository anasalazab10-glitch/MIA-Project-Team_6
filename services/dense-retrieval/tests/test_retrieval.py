import json

from src.chunking import create_chunks
from src.embeddings import EmbeddingModel
from src.dense_retrieval import DenseRetriever
from src.schemas import Candidate, RetrievalMethod
from src.vector_store import VectorStore


def test_dense_retrieval():
    # 1. Load processed elements
    with open(
        "data/mock_processed.json",
        "r",
        encoding="utf-8",
    ) as f:
        elements = json.load(f)

    # 2. Create chunks

    # Use the standard chunking configuration
    chunks = create_chunks(elements)

    print(f"\nGenerated chunks: {len(chunks)}")

    assert len(chunks) > 0

    # 3. Create embedding model
    embedding_model = EmbeddingModel()

    # 4. Embed chunks
    embeddings = embedding_model.encode_chunks(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384

   
    # 5. Create vector store
    
    vector_store = VectorStore(
        collection_name="dense_retrieval_test",
        vector_size=384,
    )

    # 6. Index chunks
    
    vector_store.add_chunks(
        chunks,
        embeddings,
    )

    # 7. Create dense retriever

    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

  
    # 8. Perform retrieval
    

    query = "What was the company's revenue in 2024?"

    response = retriever.retrieve(
        query=query,
        top_k=5,
    )

    # 9. Validate response
  

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL RESULTS")
    print("=" * 70)

    print("Query:", response.query)
    print("Method:", response.retrieval_method)
    print("Candidates:", len(response.candidates))

    assert response.query == query
    assert response.retrieval_method == RetrievalMethod.DENSE

    # Qdrant cannot return more candidates than available chunks
    assert len(response.candidates) == min(5, len(chunks))

 
    # 10. Check every candidate
  
    for candidate in response.candidates:

        assert isinstance(candidate, Candidate)

        assert candidate.retrieval_method == RetrievalMethod.DENSE

        assert candidate.rank >= 1

        assert -1 <= candidate.score <= 1

        print("\n" + "-" * 70)
        print("Rank:", candidate.rank)
        print("Score:", candidate.score)
        print("Chunk:", candidate.chunk.chunk_id)
        print("Document:", candidate.chunk.document_id)
        print("Page:", candidate.chunk.page)
        print("Section:", candidate.chunk.section)
        print("Content:", candidate.chunk.content)


    # 11. Check ranking
  

    ranks = [
        candidate.rank
        for candidate in response.candidates
    ]

    assert ranks == list(
        range(1, len(response.candidates) + 1)
    )

    # 12. Check retrieval relevance
    retrieved_text = " ".join(
        candidate.chunk.to_text()
        for candidate in response.candidates
    )

    assert "$120 million" in retrieved_text



    print("\n" + "=" * 70)
    print("Dense retrieval test passed successfully!")
    print(f"Candidates returned: {len(response.candidates)}")
    print("=" * 70)


if __name__ == "__main__":
    test_dense_retrieval()