# Retrieval API

The project includes a local FastAPI service that loads the exported FiQA
`IndexIVFPQ` artifact and returns ranked documents.

## Features

- `GET /health`: confirms artifact and model loading.
- `POST /search`: retrieves top-k documents for one query.
- `POST /batch-search`: encodes all submitted queries together and sends one
  matrix search request to Faiss. This is a true micro-batch path, not a loop
  around single-query retrieval.

## Artifact contract

```text
artifacts/fiqa_ivfpq_m96/
├── index.faiss            # tracked in Git
├── service_config.json    # tracked in Git
├── doc_ids.json           # tracked in Git; preserves index-row order
└── documents.jsonl        # generated locally; ignored by Git
```

`documents.jsonl` is not stored in Git because it is a reconstructable 45 MB
copy of FiQA metadata.

## Environment setup on macOS Apple Silicon

Use conda-forge for Faiss. This avoids mixing incompatible native OpenMP/Faiss
binaries from Conda and pip.

```bash
conda env create -f environment.yml
conda activate rag-api
pip install -r requirements-api.txt
```

If the environment already exists:

```bash
conda activate rag-api
conda install -c conda-forge faiss-cpu
pip install -r requirements-api.txt
```

## Generate local metadata

```bash
python scripts/prepare_fiqa_documents.py
```

The script downloads FiQA on first use and writes
`artifacts/fiqa_ivfpq_m96/documents.jsonl` in the exact vector-addition order
preserved by `doc_ids.json`.

## Run the API

```bash
uvicorn app.main:app
```

Open interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Do not use `--reload` from the repository root unless you explicitly exclude
`.venv/` or other environment directories; file watchers can trigger repeated
reloads while dependencies are changing.

## Example requests

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a dividend stock?","top_k":5,"nprobe":16}'
```

```bash
curl -X POST http://127.0.0.1:8000/batch-search \
  -H "Content-Type: application/json" \
  -d '{"queries":["What is a dividend stock?","How does inflation affect bond prices?"],"top_k":3,"nprobe":16}'
```

## Scope

This is a retrieval component, not a complete generative RAG application. It
returns ranked FiQA documents and similarity scores. Later extensions can add
reranking, prompt construction, an LLM, observability, and a Linux/NVIDIA GPU
serving image.
