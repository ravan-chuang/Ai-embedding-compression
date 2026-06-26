# Testing and Continuous Integration

The repository includes offline unit tests and a GitHub Actions workflow.

## Local test run

Use the production-like Conda environment locally. Faiss is installed through
conda-forge, and the development requirements intentionally do **not** install
a second pip Faiss package.

```bash
conda env create -f environment.yml
conda activate rag-api
pip install -r requirements-dev.txt
pytest -q
```

The tests are intentionally small and do not download FiQA or an embedding model.

## What is tested

- Retrieval artifact contract: the serialized Faiss index count matches the tracked
  `doc_ids.json` ordering.
- Retriever loading, ranking, batch matrix search, and index/document mismatch handling.
- FastAPI endpoint functions for health, single-query retrieval, and batch retrieval.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests.

GitHub Actions uses `environment-ci.yml`, a minimal Python-only Conda
environment, then installs one Linux `faiss-cpu` wheel from pip via
`requirements-ci.txt`. This is deliberate: the previous conda-forge Faiss
package on the hosted Linux runner failed to load because an MKL shared library
was unavailable. The CI environment is isolated and never mixes pip Faiss with
a Conda Faiss package.

The workflow:

1. creates the minimal CI environment,
2. installs CI dependencies,
3. executes `pytest -q`.

The workflow validates the CPU retrieval code path only. GPU benchmark execution
remains in the Colab notebook because GitHub-hosted runners do not provide the
NVIDIA runtime used for the benchmark.
