# Ledger – Dense Retrieval

## What I Worked On

I implemented the **Dense Retrieval** part of the Ledger project.

The main goal was to take the processed documents, split them into meaningful chunks, generate embeddings for those chunks, store them in Qdrant, and retrieve the most relevant chunks for a query.

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

The document information such as `document_id`, `page`, `section`, and `content_type` is preserved in each chunk.

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

The chunks and their embeddings are inserted into Qdrant, and the query embedding is used to search for the most similar chunks.

Currently I use:

`QdrantClient(":memory:")`

so Qdrant is running in memory for development/testing. Since the project will be containerized, we can later move Qdrant to a Docker Compose service.

## 4. Dense Retrieval

File: `src/retrieval.py`

I implemented `DenseRetriever`.

It:

1. Takes the user's query.
2. Generates its embedding.
3. Searches Qdrant.
4. Gets the most similar chunks.
5. Converts the results into our common `Candidate` format.
6. Returns the retrieved candidates with their scores and ranks.

## 5. Testing

I created and tested:

* `tests/test_embeddings.py`
* `tests/test_vector_store.py`
* `tests/test_retrieval.py`

The dense retrieval test uses:

`What was the company's revenue in 2024?`

It retrieves the top 5 results and successfully finds relevant chunks containing the **$120M 2024 revenue** information.

All three tests are currently passing.

## Current Status

Dense Retrieval is **implemented and tested successfully**.

