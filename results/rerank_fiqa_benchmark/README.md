# FiQA Reranker Evaluation

This directory records an initial 100-query FiQA evaluation of the optional
`BAAI/bge-reranker-base` CPU cross-encoder integration.

## Result

The tested reranker configuration did not improve the OPQ-IVF-PQ baseline.

| Pipeline | Recall@10 | MRR@10 | nDCG@10 | P95 latency |
|---|---:|---:|---:|---:|
| OPQ-IVF-PQ only | 0.4782 | 0.4888 | 0.4216 | 6.94 ms |
| OPQ-IVF-PQ + rerank, candidate_k=10 | 0.4782 | 0.4404 | 0.3939 | 651.84 ms |
| OPQ-IVF-PQ + rerank, candidate_k=20 | 0.4639 | 0.4390 | 0.3840 | 1509.23 ms |

## Interpretation

- At candidate_k=10, reranking preserves the candidate set but degrades MRR@10
  and nDCG@10, indicating that this reranker does not improve ordering for the
  evaluated FiQA subset.
- At candidate_k=20, the reranker also reduces Recall@10 because relevant
  candidates are pushed below the final top-10 cutoff.
- CPU reranking adds substantial latency relative to OPQ-IVF-PQ ANN retrieval.

The feature remains available for experimentation but is disabled by default.
Future work should compare domain-appropriate rerankers, document truncation
policies, and batched reranking before any production-default claim.
