# Local Retrieval API Benchmark

## Scope

This report measures end-to-end local HTTP latency. Each request includes JSON
serialization, HTTP transport, query embedding, Faiss search, and response assembly.
It is not directly comparable with the GPU-only Faiss serving benchmark in the main README.

## Run configuration

- Generated (UTC): `2026-06-26T19:34:57+00:00`
- Base URL: `http://127.0.0.1:8000`
- Python: `3.11.15`
- Platform: `macOS-26.5.1-arm64-arm-64bit`
- Index type: `IndexIVFPQ`
- Document count: `57638`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Warm-up requests per case: `5`
- Measured requests per case: `30`
- top_k: `5`
- nprobe: `16`

## Results

| Endpoint | Batch size | Requests | Client P50 | Client P95 | Server P50 | Server P95 | Query throughput |
|:--|--:|--:|--:|--:|--:|--:|--:|
| /batch-search | 8 | 30 | 9.930 ms | 11.953 ms | 8.901 ms | 10.996 ms | 797.30 q/s |
| /batch-search | 32 | 30 | 21.006 ms | 21.837 ms | 19.478 ms | 20.215 ms | 1519.05 q/s |
| /search | 1 | 30 | 6.627 ms | 7.304 ms | 5.804 ms | 6.482 ms | 149.70 q/s |

## Interpretation

- **Client latency** is the user-visible, end-to-end local request time.
- **Server latency** is the API-reported retrieval time and excludes client-side timing overhead.
- **Query throughput** is total processed queries divided by the summed client request time.
- Batch sizes above 1 use `POST /batch-search`, which encodes the complete batch and performs one matrix Faiss search call.
