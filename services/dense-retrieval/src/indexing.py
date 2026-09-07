from src.chunking import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def index_elements(elements,embedding_model: EmbeddingModel,vector_store: VectorStore,):
    """
    Convert processed document elements into retrieval chunks,
    generate embeddings, and store them in Qdrant.
    """

    if not elements:
        raise ValueError("Elements list cannot be empty.")

    # 1. Create retrieval chunks
    chunks = create_chunks(elements)

    if not chunks:
        raise ValueError(
            "No chunks were created from the provided elements."
        )

    # 2. Generate embeddings
    embeddings = embedding_model.encode_chunks(chunks)

    # 3. Store chunks and embeddings in Qdrant
    vector_store.add_chunks(chunks, embeddings)

    return chunks
