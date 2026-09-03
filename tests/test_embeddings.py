import json

from src.chunking import create_chunks
from src.embeddings import EmbeddingModel
from src.schemas import ContentType


def test_embeddings():
    # 1. Load processed elements
    with open(
        "data/mock_processed.json",
        "r",
        encoding="utf-8",
    ) as f:
        elements = json.load(f)

 
    # 2. Create chunks

    chunks = create_chunks(
        elements,
        max_chars=1000,
        overlap_chars=50,
    )

    print(f"\nGenerated chunks: {len(chunks)}")

    assert len(chunks) > 0


    # 3. Verify chunk structure

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id
        assert isinstance(chunk.page, list)
        assert all(isinstance(page, int) for page in chunk.page)
        assert chunk.content_type in ContentType

    # 4. Load embedding model

    embedding_model = EmbeddingModel()

    # 5. Test chunk -> text conversion

    for chunk in chunks[:3]:
        text = embedding_model.chunk_to_text(chunk)

        print("\nChunk:", chunk.chunk_id)
        print("Content type:", chunk.content_type)
        print("Text:", text)

        assert isinstance(text, str)
        assert len(text.strip()) > 0

        # EmbeddingModel should use Chunk.to_text()
        assert text == chunk.to_text()


    # 6. Generate chunk embeddings


    embeddings = embedding_model.encode_chunks(chunks)

    print("\nEmbedding shape:", embeddings.shape)

    # One embedding per chunk
    assert embeddings.shape[0] == len(chunks)

    # BGE-small-en-v1.5 -> 384 dimensions
    assert embeddings.shape[1] == 384

    # 7. Test query embedding

    query = "What was the company's revenue in 2024?"

    query_embedding = embedding_model.encode_query(query)

    print("Query embedding shape:", query_embedding.shape)

    assert query_embedding.shape[0] == 384



    print("\n" + "=" * 60)
    print("Embedding test passed successfully!")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    test_embeddings()