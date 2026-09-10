import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.schemas import Chunk


class VectorStore:
    def __init__(
        self,
        collection_name: str = "ledger_chunks",
        vector_size: int = 384,
        host: str | None = None,
        port: int | None = None,
    ):
        self.collection_name = collection_name

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Connect to Qdrant Cloud when Cloud credentials are provided.
        if qdrant_url and qdrant_api_key:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )

        # Otherwise, fall back to the local Qdrant instance.
        else:
            host = host or os.getenv("QDRANT_HOST", "localhost")
            port = port or int(os.getenv("QDRANT_PORT", "6333"))

            self.client = QdrantClient(
                host=host,
                port=port,
            )

        # Create the collection only if it does not already exist.
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings,
    ):
        """Store chunk embeddings and metadata in Qdrant."""

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

    def search(
        self,
        query_embedding,
        top_k: int = 30,
        document_id: str | None = None,
	content_type: str | None = None,
    ):
        """Search Qdrant for the most similar chunks."""

        must_conditions = []

        if document_id is not None:
          must_conditions.append(
          FieldCondition(
          key="document_id",
          match=MatchValue(value=document_id),
           )
         )

        if content_type is not None:
          must_conditions.append(
          FieldCondition(
          key="content_type",
          match=MatchValue(value=content_type),
           )
        )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        return results.points

    def get_all_chunks(self) -> list[Chunk]:
        """Load all indexed chunks fromQdrant."""

        chunks = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                if point.payload:
                    chunks.append(
                        Chunk.model_validate(point.payload)
                    )

            if next_offset is None:
                break

            offset = next_offset

        return chunks
