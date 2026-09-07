Docker Environment – Retrieval Service

The docker environment contains two main services: qdrant and the API. qdrant uses a Docker volume, so the indexed data is preserved and does not get lost when the container is stopped or removed


1. Build the Docker image :

cd ~/MIA-Project-Team_6
docker compose build

2. Run the application :
docker compose up api

The API runs on:

http://localhost:8000
http://localhost:6333

3. Add a new requirement :

Add the package to:
requirements.txt

Then rebuild the image:
docker compose build

Run the application again:
docker compose up api




retrieval part :
run the indexing command
docker compose run --rm api python -m <indexing_module>
Run this only when the data/chunks need to be indexed or re-indexed.

run the evaluation :
docker compose run --rm api python -m evaluation.evaluation