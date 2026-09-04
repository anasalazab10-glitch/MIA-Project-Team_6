from src.embeddings import EmbeddingModel
from src.schemas import (
    Candidate,
    Chunk,
    RetrievalMethod,
    RetrievalResponse,
)
from src.vector_store import VectorStore


DEFAULT_TOP_K = 30

class DenseRetriever:
    def __init__(self,embedding_model: EmbeddingModel,vector_store: VectorStore,):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self,query: str,top_k: int = DEFAULT_TOP_K,) -> RetrievalResponse:
        """
        Perform dense retrieval for a user query.

        1. Embed the query.
        2. Search Qdrant for similar vectors.
        3. Convert results into Candidate objects.
        4. Return a standard RetrievalResponse.
        """

        # 1. Convert the query into an embedding
        query_embedding = self.embedding_model.encode_query(query)

        # 2. Search Qdrant
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        # 3. Convert Qdrant results into Candidate objects
        candidates: list[Candidate] = []

        for rank, result in enumerate(results, start=1):
            chunk = Chunk.model_validate(result.payload)

            candidates.append(
                Candidate(
                    chunk=chunk,
                    score=float(result.score),
                    retrieval_method=RetrievalMethod.DENSE,
                    rank=rank,
                )
            )

        # 4. Return the standard retrieval response
        return RetrievalResponse(
            query=query,
            retrieval_method=RetrievalMethod.DENSE,
            candidates=candidates,
        )