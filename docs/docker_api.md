# Containerized Retrieval API

The repository ships a Docker setup for the FastAPI retrieval service.

The image uses **conda-forge Faiss** through `environment.yml`. This avoids
installing a second native Faiss runtime through pip.

## Build and run

From the repository root:

```bash
docker compose up --build
```

The first startup may take longer because the container downloads the FiQA
source metadata and the embedding model, then creates the local
`documents.jsonl` metadata file. The Faiss index and its row-order mapping are
already included in the repository artifacts.

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Check health from a second terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected fields include:

```json
{
  "status": "ok",
  "index_type": "IndexIVFPQ",
  "document_count": 57638
}
```

Stop the stack:

```bash
docker compose down
```

## Persistent FiQA cache

`docker-compose.yml` defines a named volume, `fiqa-cache`, mounted at
`/app/data`. It keeps the downloaded FiQA source corpus across container
rebuilds. The generated `documents.jsonl` file is deliberately not tracked in
Git because it is reproducible metadata derived from FiQA and `doc_ids.json`.

## Direct Docker commands

```bash
docker build -t rag-retrieval-api .
docker run --rm -p 8000:8000 rag-retrieval-api
```

## Scope

This container is a local CPU-serving demonstration. It does not expose the
GPU-only Faiss benchmark path used in the Colab experiments. The service
performs query embedding, IVF-PQ retrieval, and response assembly.
