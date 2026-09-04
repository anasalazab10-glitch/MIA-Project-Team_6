import json

from src.chunking import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def index_documents(
    input_path: str,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
):
    """
    Load processed document elements, create chunks,
    generate embeddings, and store them in Qdrant.
    """

    # 1. Load processed document elements
    with open(input_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    # 2. Create retrieval chunks
    chunks = create_chunks(elements)

    if not chunks:
        raise ValueError("No chunks were created from the input data.")

    # 3. Generate embeddings
    embeddings = embedding_model.encode_chunks(chunks)

    # 4. Store chunks and embeddings in Qdrant
    vector_store.add_chunks(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks successfully.")
    
    results = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    
    points = results[0]
    
    print(f"Qdrant contains {len(points)} points:")
    
    for point in points:
        print(
            f"- {point.payload['chunk_id']} | "
            f"document={point.payload['document_id']} | "
            f"page={point.payload['page']} | "
            f"type={point.payload['content_type']}"
        )

    return chunks


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()

    chunks = index_documents(
        input_path="data/mock_processed.json",
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    print(f"Indexed {len(chunks)} chunks successfully.")