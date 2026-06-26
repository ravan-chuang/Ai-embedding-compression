# Retrieval API

This folder turns the benchmark into a small, CPU-portable retrieval service.

## Why CPU Faiss for the first service version?

The benchmark evaluates GPU Faiss IVF-PQ ADC in Colab. The API first exports a
**CPU-serializable** version of the selected Faiss index so it can run locally
on macOS and in a standard Docker container. The index type and compression
design remain the same, but production GPU serving requires a Linux/NVIDIA
deployment image and a GPU-specific Faiss runtime.

## Artifact contract

The service expects this directory:

```text
artifacts/fiqa_ivfpq_m96/
├── index.faiss
├── documents.jsonl
└── service_config.json
```

- `index.faiss`: CPU Faiss index created with `faiss.write_index`.
- `documents.jsonl`: one `{doc_id, title, text}` object per index row, in the
  exact order vectors were added to the index.
- `service_config.json`: embedding model and default query-time settings.

## 1. Export from the notebook

Copy `scripts/export_service_artifacts.py` into the Colab runtime or add its
contents as a notebook cell after the selected IVF-PQ index is built.

For the current FiQA notebook, export the standard `M=96, nprobe=16` IVF-PQ
index first. The external PyTorch OPQ path needs one additional exported
rotation transform, so it is intentionally not the first API artifact.

## 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

Open interactive docs at:

```text
http://127.0.0.1:8000/docs
```

## 3. Example requests

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a dividend stock?","top_k":5,"nprobe":16}'
```

## 4. Docker

After artifacts have been exported to `artifacts/fiqa_ivfpq_m96/`:

```bash
docker build -t embedding-retrieval-api .
docker run --rm -p 8000:8000 embedding-retrieval-api
```

## Current scope

This is a retrieval component, not a complete generative RAG application. It
returns ranked FiQA source documents and scores. A later service layer can add
reranking, prompt construction, an LLM, observability, request batching, and a
Linux/NVIDIA GPU deployment path.
