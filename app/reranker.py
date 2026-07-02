"""Cross-encoder reranking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    index: int
    score: float


class CrossEncoderReranker:
    """Optional second-stage query-document reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: CrossEncoder | None = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
        )

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Reranker is not loaded.")

        if not documents:
            return []

        pairs = [
            (
                query,
                self._document_text(document),
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = []
        for document, score in zip(documents, scores):
            enriched = dict(document)
            enriched["ann_score"] = enriched["score"]
            enriched["rerank_score"] = float(score)
            ranked.append(enriched)

        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        return ranked

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        title = document.get("title", "").strip()
        text = document.get("text", "").strip()

        if title and text:
            return f"{title}\n{text}"
        return title or text