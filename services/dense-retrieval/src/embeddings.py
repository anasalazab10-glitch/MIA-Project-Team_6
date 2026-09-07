from sentence_transformers import SentenceTransformer

from src.schemas import Chunk


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

class EmbeddingModel:
    def __init__(self,model_name: str = DEFAULT_MODEL_NAME,):
        self.model = SentenceTransformer(model_name)

    def chunk_to_text(self, chunk: Chunk) -> str:
        """Convert a chunk into text for embedding."""
        return chunk.to_text()

    def encode_chunks(self, chunks: list[Chunk]):
        """Generate embeddings for a list of chunks."""
        texts = [self.chunk_to_text(chunk) for chunk in chunks]

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_query(self, query: str):
        """Generate an embedding for a search query."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )