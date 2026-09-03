from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import uuid

from src.schemas import Chunk


class VectorStore:
    def __init__(self,collection_name: str = "ledger_chunks",vector_size: int = 384,):
        self.collection_name = collection_name

        # Local in-memory Qdrant.
        # Data exists while this Python process is running.
        self.client = QdrantClient(":memory:")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def add_chunks(self, chunks: list[Chunk], embeddings, ):
        """
        Store chunk embeddings and their metadata in Qdrant.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            chunk.chunk_id,
                        )
                    ),
                    vector=embedding.tolist(),
                    payload=chunk.model_dump(mode="json"),
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self,query_embedding,top_k: int = 30,):
        """
        Search Qdrant for the most similar chunks.
        """

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
        )

        return results.points