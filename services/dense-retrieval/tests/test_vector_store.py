import json

from src.chunking import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def test_vector_store():

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

    # 3. Generate embeddings

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode_chunks(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384

  
    # 4. Create vector store


    vector_store = VectorStore(
        collection_name="test_chunks",
        vector_size=384,
    )

    # 5. Store chunks

    vector_store.add_chunks(
        chunks,
        embeddings,
    )

    print("Chunks successfully stored in Qdrant.")


    # 6. Search

    query = "What was the company's revenue in 2024?"

    query_embedding = embedding_model.encode_query(query)

    results = vector_store.search(
        query_embedding,
        top_k=5,
    )

    print(f"\nResults returned: {len(results)}")

   
    # 7. Validate results
   

    assert len(results) == min(5, len(chunks))

    for result in results:

        print("\n" + "=" * 60)
        print("Score:", result.score)

        payload = result.payload

        print("Chunk ID:", payload["chunk_id"])
        print("Document:", payload["document_id"])
        print("Page:", payload["page"])
        print("Section:", payload["section"])
        print("Content:", payload["content"])

        assert "chunk_id" in payload
        assert "document_id" in payload
        assert "page" in payload
        assert "section" in payload
        assert "content" in payload

        assert isinstance(payload["page"], list)

    print("\n" + "=" * 60)
    print("Vector store test passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_vector_store()