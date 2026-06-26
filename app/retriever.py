"""Faiss-backed retrieval service primitives."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalService:
    """Loads a serialized CPU Faiss index, document metadata, and embedding model."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.index: faiss.Index | None = None
        self.model: SentenceTransformer | None = None
        self.documents: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.model is not None and bool(self.documents)

    def load(self) -> None:
        config_path = self.artifact_dir / "service_config.json"
        index_path = self.artifact_dir / "index.faiss"
        documents_path = self.artifact_dir / "documents.jsonl"

        missing = [str(p) for p in (config_path, index_path, documents_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing retrieval artifacts. Export them from the notebook first: "
                + ", ".join(missing)
            )

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_path))

        with documents_path.open("r", encoding="utf-8") as f:
            self.documents = [json.loads(line) for line in f if line.strip()]

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                f"Index/document mismatch: index contains {self.index.ntotal} vectors "
                f"but documents.jsonl contains {len(self.documents)} rows."
            )

        model_name = self.config["embedding_model"]
        self.model = SentenceTransformer(model_name, device="cpu")

        default_nprobe = self.config.get("default_nprobe")
        if default_nprobe is not None:
            self.set_nprobe(int(default_nprobe))

    def set_nprobe(self, nprobe: int) -> None:
        if self.index is None:
            raise RuntimeError("Retriever is not loaded.")

        try:
            ivf = faiss.extract_index_ivf(self.index)
            ivf.nprobe = nprobe
        except RuntimeError:
            # Exact Flat indexes do not expose nprobe. Leave them unchanged.
            pass

    def search(self, query: str, top_k: int, nprobe: int | None = None) -> dict[str, Any]:
        if not self.is_ready or self.index is None or self.model is None:
            raise RuntimeError("Retriever is not ready.")

        if nprobe is not None:
            self.set_nprobe(nprobe)

        start = time.perf_counter()
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = self.index.search(vector, top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            document = self.documents[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "doc_id": document["doc_id"],
                    "title": document.get("title", ""),
                    "text": document.get("text", ""),
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "nprobe": nprobe if nprobe is not None else self.config.get("default_nprobe"),
            "latency_ms": round(latency_ms, 3),
            "index_type": self.config.get("index_type", type(self.index).__name__),
            "results": results,
        }
