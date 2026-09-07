# Ledger – Retrieval System


I implemented and connected the Retrieval part of the Ledger project.

The main goal was to take the processed documents, split them into meaningful chunks, generate embeddings, store them in Qdrant, retrieve relevant candidates using multiple retrieval methods, and rerank them before sending the final candidates to the Reasoning system.

## 1. Chunking

File: `src/chunking.py`

I implemented:

* Section-aware chunking
* Table-aware chunking
* Sentence-based text splitting

Current settings:

* Maximum chunk size: **1000 characters**
* Overlap: **150 characters**

Tables are kept as separate chunks, while text is split based on sentences.

The current mock data contains **27 elements**, which are converted into **12 chunks**:

* 8 text chunks
* 4 table chunks

## 2. Embeddings

File: `src/embeddings.py`

I used **Sentence Transformers** with:

`BAAI/bge-small-en-v1.5`

The model generates **384-dimensional embeddings**.

Both document chunks and user queries are embedded using the same model.

For tables, the structured table is converted to text before generating its embedding.

## 3. Qdrant

File: `src/vector_store.py`

I connected the embeddings to **Qdrant** for vector storage and similarity search.

Qdrant uses:

* Vector size: **384**
* Distance: **Cosine similarity**

Qdrant is now running as a Docker Compose service on port:

`6333`

The Qdrant data is stored using a Docker volume so that the indexed data is preserved when the container is stopped or removed.

## 4. Retrieval Pipeline

The retrieval pipeline combines multiple retrieval methods to improve the quality of the retrieved candidates.

The pipeline works as follows:

1. The user sends a query to the Retrieval API.
2. Dense Retrieval searches Qdrant using the query embedding.
3. BM25 performs keyword-based retrieval.
4. The Dense and BM25 results are combined using Hybrid RRF.
5. The system over-retrieves up to **30 candidates**.
6. A Cross-Encoder reranks these candidates.
7. The final **top 5 candidates** are returned.

The API endpoint is:

`POST http://localhost:8000/search`

The API runs on port:

`8000`

## 5. Retrieval and Reasoning Integration

The Retrieval and Reasoning systems work together as two separate parts.

The Retrieval system is responsible for finding the most relevant evidence, while the Reasoning system uses this evidence to understand the question and produce the final answer.

The overall flow is:

```text
User Question
      ↓
Reasoning System
      ↓
Retrieval API
      ↓
Dense + BM25
      ↓
Hybrid RRF
      ↓
Top 30 Candidates
      ↓
Cross-Encoder Reranking
      ↓
Final Top 5 Candidates
      ↓
Reasoning System
      ↓
Final Answer + Supporting Evidence
```

The Reasoning system sends the user's question to the Retrieval API through `/search`.

The Retrieval API returns the final 5 reranked candidates. The Reasoning system then uses these candidates as evidence for its reasoning and final response.

## 6. Indexing

Indexing is used to prepare the current document chunks for retrieval and store them in Qdrant.

Run the indexing command when the data or chunks need to be indexed or re-indexed:

```bash
docker compose run --rm api python -m <indexing_module>
```

The indexed data is stored in the Qdrant Docker volume.

## 7. Evaluation

I created a new `evaluation/` folder and a new evaluation dataset for the current corpus.

The previous evaluation dataset was based on **27 chunks**, while the current indexed corpus contains **12 chunks**, so the previous evaluation was not compatible with the current data.

The previous evaluation files were not deleted.

The current evaluation compares:

* Dense Retrieval
* BM25
* Hybrid RRF
* Hybrid RRF + Cross-Encoder

Run the evaluation using:

```bash
docker compose run --rm api python -m evaluation.evaluation
```

Current evaluation results:

| Method                       |  Hit@K | Recall@K | Precision@K |    MRR | Avg. Latency |
| ---------------------------- | -----: | -------: | ----------: | -----: | -----------: |
| Dense Top 30                 | 1.0000 |   1.0000 |      0.0583 | 0.8958 |     18.60 ms |
| BM25 Top 30                  | 1.0000 |   1.0000 |      0.0583 | 0.9167 |      0.20 ms |
| Hybrid RRF Top 30            | 1.0000 |   1.0000 |      0.0583 | 0.8958 |     17.81 ms |
| Hybrid + Cross-Encoder Top 5 | 1.0000 |   0.8958 |      0.3000 | 0.9375 |    135.98 ms |

The evaluation shows that the retrieval methods successfully find relevant candidates within the top 30. The Cross-Encoder improves the ranking quality when reducing the results to the final 5 candidates, with an increase in latency.

## Current Status

The Retrieval system is **implemented, tested, evaluated, Dockerized, and connected to the Reasoning system through the Retrieval API**.
