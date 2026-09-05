# LEDGER – Project Structure & Docker Guide

## 1. Project Structure

```text
MIA-Project-Team_6/
├── docker-compose.yml
├── services/
│   ├── dense-retrieval/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   ├── data/
│   │   └── evaluation/
│   │
│   ├── doc-processor-api/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       └── main.py
│   │
│   ├── reasoning/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   │
│   ├── validator-service/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   │
│   ├── orchestrator/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   │
│   └── ui/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app.py
```

Each service has its own `Dockerfile` and `requirements.txt`.

---

## 2. Service Ports

| Service         | Local URL               |
| --------------- | ----------------------- |
| Dense Retrieval | `http://localhost:8000` |
| Reasoning       | `http://localhost:8001` |
| Doc Processor   | `http://localhost:8002` |
| Orchestrator    | `http://localhost:8003` |
| Validator       | `http://localhost:8004` |
| UI              | `http://localhost:7860` |
| Qdrant          | `http://localhost:6333` |

Inside Docker Compose, services should communicate using their **service names**, for example:

```text
http://dense-retrieval:8000
http://reasoning:8000
http://doc-processor:8000
http://validator:8000
http://orchestrator:8000
```

Do not use `localhost` for communication between containers.

---

## 3. Dense Retrieval

### Run the retrieval API

From the project root:

```bash
docker compose up dense-retrieval
```

### Run indexing

Use:

```bash
docker compose run --rm dense-retrieval python -m <indexing_module>
```

Replace `<indexing_module>` with the actual indexing module.

### Run evaluation

```bash
docker compose run --rm dense-retrieval python -m evaluation.evaluation
```

Qdrant runs automatically with the project:

```text
Qdrant → http://localhost:6333
```

---

## 4. Orchestrator

The orchestrator is the main backend that connects the services.

It should:

1. Receive the request from the UI.
2. Call the document processor when a document needs processing.
3. Send retrieval requests to Dense Retrieval.
4. Send the retrieved context to Reasoning.
5. Send the generated answer/evidence to the Validator.
6. Return the final result to the UI.

The orchestrator's application code should be placed in:

```text
services/orchestrator/main.py
```

Its Dockerfile and requirements should stay in the same folder.

---

## 5. UI

The UI should be placed in:

```text
services/ui/
├── Dockerfile
├── requirements.txt
└── app.py
```

The Gradio application should be in:

```text
services/ui/app.py
```

The UI should communicate with the **Orchestrator**, not directly with every backend service.

Inside Docker, use:

```text
http://orchestrator:8000
```

---

## 6. Run the Whole Project

From the project root:

```bash
docker compose build
```

Then:

```bash
docker compose up
```

Or build and start together:

```bash
docker compose up --build
```

Check running containers:

```bash
docker compose ps
```

Open the UI:

```text
http://localhost:7860
```

---

## 7. Adding a New Requirement

If you add a Python package to a service:

1. Add it to that service's `requirements.txt`.
2. Rebuild that service.

For example:

```bash
docker compose build dense-retrieval
```

If you want to rebuild everything:

```bash
docker compose build
```

Then restart:

```bash
docker compose up
```

Do not install new packages manually inside the running container. Add them to `requirements.txt` so everyone gets the same environment.

---

## 8. After Changing Code

For one service:

```bash
docker compose up --build <service-name>
```

For the whole project:

```bash
docker compose up --build
```

The final `docker-compose.yml` should stay at the **project root**:

```text
MIA-Project-Team_6/docker-compose.yml
```
