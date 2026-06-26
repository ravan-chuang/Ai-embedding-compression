"""
Run this script *inside the notebook runtime* after building a CPU-serializable
Faiss IVF-PQ index. It exports the artifact bundle consumed by the FastAPI service.

Required notebook variables:
  - index_to_export: a CPU Faiss index, or a GPU Faiss index convertible to CPU
  - doc_ids: list[str]
  - doc_texts: list[str]
  - EMBEDDING_MODEL: string
Optional:
  - doc_titles: list[str]
  - DEFAULT_NPROBE: int (defaults to 16)

Example:
  index_to_export = faiss.index_gpu_to_cpu(gpu_ivfpq_index)
  %run scripts/export_service_artifacts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss

EXPORT_DIR = Path("service_artifacts/fiqa_ivfpq_m96")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

try:
    index_to_export
except NameError as exc:
    raise RuntimeError(
        "Set `index_to_export` first. For a GPU index use "
        "`index_to_export = faiss.index_gpu_to_cpu(gpu_ivfpq_index)`."
    ) from exc

try:
    doc_titles
except NameError:
    doc_titles = [""] * len(doc_ids)

if not (len(doc_ids) == len(doc_texts) == len(doc_titles) == index_to_export.ntotal):
    raise ValueError("Document metadata lengths must exactly match index_to_export.ntotal.")

faiss.write_index(index_to_export, str(EXPORT_DIR / "index.faiss"))

with (EXPORT_DIR / "documents.jsonl").open("w", encoding="utf-8") as f:
    for doc_id, title, text in zip(doc_ids, doc_titles, doc_texts):
        f.write(json.dumps(
            {"doc_id": str(doc_id), "title": str(title), "text": str(text)},
            ensure_ascii=False,
        ) + "\n")

config = {
    "embedding_model": EMBEDDING_MODEL,
    "index_type": type(index_to_export).__name__,
    "vector_count": int(index_to_export.ntotal),
    "dimension": int(index_to_export.d),
    "default_nprobe": int(globals().get("DEFAULT_NPROBE", 16)),
    "normalization": "L2-normalized embeddings; inner-product search",
}
(EXPORT_DIR / "service_config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Exported service artifacts to: {EXPORT_DIR.resolve()}")
