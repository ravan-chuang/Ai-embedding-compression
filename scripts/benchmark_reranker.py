"""Evaluate OPQ-IVF-PQ retrieval with optional cross-encoder reranking on FiQA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from beir import util
from app.retriever import RetrievalService

FIQA_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/fiqa.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FiQA OPQ-IVF-PQ retrieval with optional "
            "BGE cross-encoder reranking."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/fiqa_opq_ivfpq_m96",
    )
    parser.add_argument(
        "--data-dir",
        default=".cache/beir",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rerank_fiqa_benchmark",
    )
    parser.add_argument(
        "--candidate-ks",
        nargs="+",
        type=int,
        default=[10, 20],
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional deterministic subset for smoke tests.",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=2,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            items[str(item["_id"])] = item

    return items


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}

    with path.open("r", encoding="utf-8") as file:
        header = next(file).strip().split("\t")

        if header[:3] != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrels header: {header}")

        for line in file:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    return qrels


def ranking_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    top_k: int,
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for query_id, ranked_doc_ids in rankings.items():
        relevance = qrels[query_id]
        relevant_doc_ids = {
            doc_id
            for doc_id, score in relevance.items()
            if score > 0
        }

        retrieved = ranked_doc_ids[:top_k]
        recalls.append(
            len(set(retrieved) & relevant_doc_ids) / len(relevant_doc_ids)
            if relevant_doc_ids
            else 0.0
        )

        first_relevant_rank = next(
            (
                rank
                for rank, doc_id in enumerate(retrieved, start=1)
                if relevance.get(doc_id, 0) > 0
            ),
            None,
        )
        reciprocal_ranks.append(
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        dcg = sum(
            (2 ** relevance.get(doc_id, 0) - 1) / math.log2(rank + 1)
            for rank, doc_id in enumerate(retrieved, start=1)
        )

        ideal_scores = sorted(relevance.values(), reverse=True)[:top_k]
        idcg = sum(
            (2 ** score - 1) / math.log2(rank + 1)
            for rank, score in enumerate(ideal_scores, start=1)
        )
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "recall_at_10": float(np.mean(recalls)),
        "mrr_at_10": float(np.mean(reciprocal_ranks)),
        "ndcg_at_10": float(np.mean(ndcgs)),
    }


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def evaluate_pipeline(
    service: RetrievalService,
    query_items: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    *,
    top_k: int,
    candidate_k: int,
    nprobe: int,
    rerank: bool,
) -> dict[str, Any]:
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    embedding_latencies: list[float] = []
    ann_latencies: list[float] = []
    rerank_latencies: list[float] = []

    for query_id, query_text in query_items:
        response = service.search(
            query=query_text,
            top_k=top_k,
            nprobe=nprobe,
            rerank=rerank,
            candidate_k=candidate_k,
        )

        rankings[query_id] = [
            str(item["doc_id"])
            for item in response["results"]
        ]

        latencies.append(float(response["latency_ms"]))

        # search() returns per-query result fields plus total latency.
        # Component timing is returned by search_many(), so use zero here
        # unless a later service API exposes it per item.
        embedding_latencies.append(0.0)
        ann_latencies.append(0.0)
        rerank_latencies.append(0.0)

    metrics = ranking_metrics(rankings, qrels, top_k=top_k)

    return {
        "rerank_enabled": rerank,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "nprobe": nprobe,
        "query_count": len(query_items),
        **metrics,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "timing_scope": (
            "Sequential in-process RetrievalService calls; includes "
            "query embedding, OPQ rotation, Faiss search, optional "
            "CPU reranking, and result formatting."
        ),
        "embedding_latency_note": (
            "Per-component timings are not emitted by search(); "
            "use service batch instrumentation for component breakdown."
        ),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# FiQA Two-Stage Retrieval Benchmark",
        "",
        "| Pipeline | Candidate K | Recall@10 | MRR@10 | nDCG@10 | Mean latency | P50 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        pipeline = (
            "OPQ-IVF-PQ + BGE rerank"
            if row["rerank_enabled"]
            else "OPQ-IVF-PQ only"
        )
        lines.append(
            "| "
            f"{pipeline} | "
            f"{row['candidate_k']} | "
            f"{row['recall_at_10']:.4f} | "
            f"{row['mrr_at_10']:.4f} | "
            f"{row['ndcg_at_10']:.4f} | "
            f"{row['mean_latency_ms']:.2f} ms | "
            f"{row['p50_latency_ms']:.2f} ms | "
            f"{row['p95_latency_ms']:.2f} ms |"
        )

    lines.extend(
        [
            "",
            "## Timing notes",
            "",
            "- Timing is local CPU service-side execution, not GPU-only Faiss timing.",
            "- It includes embedding, OPQ query rotation, ANN retrieval, optional cross-encoder reranking, and result formatting.",
            "- It excludes HTTP transport, Docker startup, and model-download time.",
            "- Reranking can improve MRR/nDCG only when a relevant document is already present in the ANN candidate pool.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    if any(candidate_k < args.top_k for candidate_k in args.candidate_ks):
        raise ValueError("Every --candidate-ks value must be >= --top-k.")

    data_root = Path(args.data_dir)
    dataset_path = Path(util.download_and_unzip(FIQA_URL, str(data_root)))

    queries = read_jsonl(dataset_path / "queries.jsonl")
    qrels = read_qrels(dataset_path / "qrels" / "test.tsv")

    query_items = [
        (query_id, queries[query_id]["text"])
        for query_id in sorted(qrels)
        if query_id in queries
    ]

    if args.max_queries is not None:
        query_items = query_items[:args.max_queries]

    if not query_items:
        raise RuntimeError("No FiQA queries with qrels were loaded.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    service = RetrievalService(args.artifact_dir)
    service.load()

    warmup_count = min(args.warmup_queries, len(query_items))

    for _, query_text in query_items[:warmup_count]:
        service.search(
            query=query_text,
            top_k=args.top_k,
            nprobe=args.nprobe,
            rerank=False,
            candidate_k=args.top_k,
        )
        service.search(
            query=query_text,
            top_k=args.top_k,
            nprobe=args.nprobe,
            rerank=True,
            candidate_k=max(args.candidate_ks),
        )

    rows: list[dict[str, Any]] = []

    rows.append(
        evaluate_pipeline(
            service,
            query_items,
            qrels,
            top_k=args.top_k,
            candidate_k=args.top_k,
            nprobe=args.nprobe,
            rerank=False,
        )
    )

    for candidate_k in args.candidate_ks:
        rows.append(
            evaluate_pipeline(
                service,
                query_items,
                qrels,
                top_k=args.top_k,
                candidate_k=candidate_k,
                nprobe=args.nprobe,
                rerank=True,
            )
        )

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, rows)

    print(f"Dataset: {dataset_path}")
    print(f"Queries evaluated: {len(query_items)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {markdown_path}")

    for row in rows:
        label = (
            f"rerank_k{row['candidate_k']}"
            if row["rerank_enabled"]
            else "ann_only"
        )
        print(
            f"{label:>12} | "
            f"R@10={row['recall_at_10']:.4f} | "
            f"MRR@10={row['mrr_at_10']:.4f} | "
            f"nDCG@10={row['ndcg_at_10']:.4f} | "
            f"P95={row['p95_latency_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
