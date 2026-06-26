"""Benchmark the local FastAPI retrieval service over real HTTP requests.

Measures end-to-end client latency for /search and /batch-search, including
HTTP serialization, query embedding, Faiss retrieval, and response assembly.
It writes raw request measurements plus a compact summary and Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_QUERIES = [
    "What is a dividend stock?",
    "How does inflation affect bond prices?",
    "What is the difference between an ETF and a mutual fund?",
    "How do interest rates affect stock valuations?",
    "What is dollar cost averaging?",
    "What is the difference between a bond and a stock?",
    "How does compound interest work?",
    "What is a credit default swap?",
    "What is an index fund?",
    "How do stock splits work?",
    "What is a recession?",
    "What does market capitalization mean?",
    "What is a balance sheet?",
    "How are capital gains taxed?",
    "What is a 401(k)?",
    "What is a yield curve?",
    "What is the difference between revenue and profit?",
    "How do dividends affect stock prices?",
    "What is a limit order?",
    "What is a short sale?",
    "What is diversification?",
    "What does P/E ratio mean?",
    "What is quantitative easing?",
    "How do treasury bonds work?",
    "What is an IPO?",
    "What is liquidity risk?",
    "What is a bear market?",
    "How does inflation affect savings?",
    "What is a hedge fund?",
    "What is a market correction?",
    "What is free cash flow?",
    "What is a credit score?",
]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile without NumPy."""
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            if response.status != 200:
                raise RuntimeError(f"{path} returned HTTP {response.status}: {body[:500]}")
            return json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach {base_url}. Start the service with `uvicorn app.main:app` first."
        ) from exc


def get_json(base_url: str, path: str, timeout_s: float) -> dict[str, Any]:
    try:
        with urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            if response.status != 200:
                raise RuntimeError(f"{path} returned HTTP {response.status}: {body[:500]}")
            return json.loads(body)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"Could not reach {base_url}. Start the service with `uvicorn app.main:app` first."
        ) from exc


def build_batch(queries: list[str], batch_size: int, offset: int) -> list[str]:
    return [queries[(offset + i) % len(queries)] for i in range(batch_size)]


def server_latency_ms(response: dict[str, Any], endpoint: str) -> float | None:
    if endpoint == "/search":
        value = response.get("latency_ms")
    else:
        value = response.get("latency_ms_total")
    return float(value) if value is not None else None


def run_case(
    *,
    base_url: str,
    endpoint: str,
    batch_size: int,
    queries: list[str],
    warmup: int,
    runs: int,
    top_k: int,
    nprobe: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    def payload_for(iteration: int) -> dict[str, Any]:
        selected = build_batch(queries, batch_size, iteration * batch_size)
        if endpoint == "/search":
            return {"query": selected[0], "top_k": top_k, "nprobe": nprobe}
        return {"queries": selected, "top_k": top_k, "nprobe": nprobe}

    for i in range(warmup):
        post_json(base_url, endpoint, payload_for(i), timeout_s)

    rows: list[dict[str, Any]] = []
    for i in range(runs):
        payload = payload_for(i + warmup)
        start = time.perf_counter()
        response = post_json(base_url, endpoint, payload, timeout_s)
        client_latency_ms = (time.perf_counter() - start) * 1000

        expected_results = 1 if endpoint == "/search" else batch_size
        if endpoint == "/search":
            actual_results = len(response.get("results", []))
        else:
            actual_results = len(response.get("items", []))
        if endpoint == "/batch-search" and actual_results != expected_results:
            raise RuntimeError(
                f"Batch response mismatch: expected {expected_results} items, got {actual_results}."
            )

        rows.append(
            {
                "endpoint": endpoint,
                "batch_size": batch_size,
                "run": i + 1,
                "client_latency_ms": round(client_latency_ms, 3),
                "server_latency_ms": server_latency_ms(response, endpoint),
                "returned_items": actual_results,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["endpoint"], int(row["batch_size"])), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (endpoint, batch_size), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        client = [float(row["client_latency_ms"]) for row in group]
        server = [float(row["server_latency_ms"]) for row in group if row["server_latency_ms"] is not None]
        total_queries = len(group) * batch_size
        total_seconds = sum(client) / 1000.0
        summaries.append(
            {
                "endpoint": endpoint,
                "batch_size": batch_size,
                "requests": len(group),
                "total_queries": total_queries,
                "client_p50_ms": round(percentile(client, 0.50), 3),
                "client_p95_ms": round(percentile(client, 0.95), 3),
                "client_mean_ms": round(statistics.mean(client), 3),
                "server_p50_ms": round(percentile(server, 0.50), 3) if server else "",
                "server_p95_ms": round(percentile(server, 0.95), 3) if server else "",
                "query_throughput_qps": round(total_queries / total_seconds, 2) if total_seconds else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    base_url: str,
    health: dict[str, Any],
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Local Retrieval API Benchmark",
        "",
        "## Scope",
        "",
        "This report measures end-to-end local HTTP latency. Each request includes JSON",
        "serialization, HTTP transport, query embedding, Faiss search, and response assembly.",
        "It is not directly comparable with the GPU-only Faiss serving benchmark in the main README.",
        "",
        "## Run configuration",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Base URL: `{base_url}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Platform: `{platform.platform()}`",
        f"- Index type: `{health.get('index_type')}`",
        f"- Document count: `{health.get('document_count')}`",
        f"- Embedding model: `{health.get('embedding_model')}`",
        f"- Warm-up requests per case: `{args.warmup}`",
        f"- Measured requests per case: `{args.runs}`",
        f"- top_k: `{args.top_k}`",
        f"- nprobe: `{args.nprobe}`",
        "",
        "## Results",
        "",
        "| Endpoint | Batch size | Requests | Client P50 | Client P95 | Server P50 | Server P95 | Query throughput |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in summaries:
        lines.append(
            "| {endpoint} | {batch_size} | {requests} | {client_p50_ms:.3f} ms | "
            "{client_p95_ms:.3f} ms | {server_p50_ms} ms | {server_p95_ms} ms | "
            "{query_throughput_qps:.2f} q/s |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Client latency** is the user-visible, end-to-end local request time.",
            "- **Server latency** is the API-reported retrieval time and excludes client-side timing overhead.",
            "- **Query throughput** is total processed queries divided by the summed client request time.",
            "- Batch sizes above 1 use `POST /batch-search`, which encodes the complete batch and performs one matrix Faiss search call.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--results-dir", default="results/api_benchmark")
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs must be >= 1.")
    if any(size < 1 or size > len(DEFAULT_QUERIES) for size in args.batch_sizes):
        parser.error(f"--batch-sizes values must be between 1 and {len(DEFAULT_QUERIES)}.")
    return args


def main() -> None:
    args = parse_args()
    health = get_json(args.base_url, "/health", args.timeout_seconds)
    if health.get("status") != "ok":
        raise RuntimeError(f"API is not ready: {health}")

    all_rows: list[dict[str, Any]] = []
    print(f"API ready: {health.get('index_type')} with {health.get('document_count')} documents")

    for batch_size in args.batch_sizes:
        endpoint = "/search" if batch_size == 1 else "/batch-search"
        print(f"Benchmarking {endpoint} batch_size={batch_size} ...")
        all_rows.extend(
            run_case(
                base_url=args.base_url,
                endpoint=endpoint,
                batch_size=batch_size,
                queries=DEFAULT_QUERIES,
                warmup=args.warmup,
                runs=args.runs,
                top_k=args.top_k,
                nprobe=args.nprobe,
                timeout_s=args.timeout_seconds,
            )
        )

    summaries = summarize(all_rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "raw_requests.csv", all_rows)
    write_csv(results_dir / "summary.csv", summaries)
    write_report(
        results_dir / "api_benchmark_report.md",
        base_url=args.base_url,
        health=health,
        args=args,
        summaries=summaries,
    )

    print("\nSummary")
    for row in summaries:
        print(
            f"{row['endpoint']:13} batch={row['batch_size']:>2} "
            f"P50={row['client_p50_ms']:>7.3f} ms "
            f"P95={row['client_p95_ms']:>7.3f} ms "
            f"QPS={row['query_throughput_qps']:>8.2f}"
        )
    print(f"\nWrote results to: {results_dir}")


if __name__ == "__main__":
    main()
