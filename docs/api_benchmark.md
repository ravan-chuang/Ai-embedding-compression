# Local Retrieval API Benchmark

`scripts/benchmark_api.py` measures the running FastAPI service through real
local HTTP requests.

It reports:

- Client P50/P95 latency: JSON serialization, HTTP transport, query embedding,
  Faiss search, and response assembly.
- API-reported P50/P95 retrieval time.
- Query throughput in queries per second.
- Raw request samples and a compact Markdown report.

This is a **local CPU application benchmark**. Do not compare its timings
directly with the GPU-only Faiss serving benchmark in the repository README.

## Run

Start the API in one terminal:

```bash
conda activate rag-api
cd ~/Desktop/Ai-embedding-compression-github
uvicorn app.main:app
```

In a second terminal:

```bash
conda activate rag-api
cd ~/Desktop/Ai-embedding-compression-github
python scripts/benchmark_api.py --warmup 5 --runs 30 --batch-sizes 1 8 32
```

The script writes:

```text
results/api_benchmark/
├── raw_requests.csv
├── summary.csv
└── api_benchmark_report.md
```

## Notes

- Batch size `1` uses `/search`.
- Batch sizes greater than `1` use `/batch-search`.
- `/batch-search` embeds the complete query batch and calls Faiss once with a
  query matrix, rather than looping over the single-query endpoint.
- Keep the machine otherwise idle while benchmarking.
