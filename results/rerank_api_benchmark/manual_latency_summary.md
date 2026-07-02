# Local CPU Reranker Latency Smoke Test

Environment:
- Host: Apple Silicon macOS local API environment
- Retriever: MiniLM + OPQ-IVF-PQ
- Reranker: BAAI/bge-reranker-base on CPU
- Query: `What is a dividend stock?`
- nprobe: 16
- top_k: 5

| Pipeline | Candidate K | End-to-end API latency |
|---|---:|---:|
| OPQ-IVF-PQ only | 5 | 15.198 ms |
| OPQ-IVF-PQ + CrossEncoder rerank | 10 | 668.218 ms |
| OPQ-IVF-PQ + CrossEncoder rerank | 20 | 1225.425 ms |
| OPQ-IVF-PQ + CrossEncoder rerank | 50 | 3011.655 ms |

Notes:
- These are manual warm-request smoke-test measurements, not a statistically rigorous benchmark.
- Latency includes query embedding, OPQ query rotation, Faiss search, CPU cross-encoder reranking, and response assembly.
- Results must not be compared directly with GPU-only Faiss search latency.
- Candidate K=20 surfaced a more directly relevant dividend-stock explanation than K=10 for this query.
