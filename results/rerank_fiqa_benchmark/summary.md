# FiQA Two-Stage Retrieval Benchmark

| Pipeline | Candidate K | Recall@10 | MRR@10 | nDCG@10 | Mean latency | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPQ-IVF-PQ only | 10 | 0.4782 | 0.4888 | 0.4216 | 6.03 ms | 5.89 ms | 6.94 ms |
| OPQ-IVF-PQ + BGE rerank | 10 | 0.4782 | 0.4404 | 0.3939 | 536.97 ms | 590.87 ms | 651.84 ms |
| OPQ-IVF-PQ + BGE rerank | 20 | 0.4639 | 0.4390 | 0.3840 | 1234.02 ms | 1251.09 ms | 1509.23 ms |

## Timing notes

- Timing is local CPU service-side execution, not GPU-only Faiss timing.
- It includes embedding, OPQ query rotation, ANN retrieval, optional cross-encoder reranking, and result formatting.
- It excludes HTTP transport, Docker startup, and model-download time.
- Reranking can improve MRR/nDCG only when a relevant document is already present in the ANN candidate pool.
