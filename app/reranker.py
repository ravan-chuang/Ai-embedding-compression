"""Cross-encoder reranking primitives."""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


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
        return self.rerank_many([(query, documents)])[0]

    def rerank_many(
        self,
        query_documents: list[tuple[str, list[dict[str, Any]]]],
    ) -> list[list[dict[str, Any]]]:
        """Rerank candidate lists using one CrossEncoder batch prediction."""

        if self.model is None:
            raise RuntimeError("Reranker is not loaded.")

        flat_pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []

        for query, documents in query_documents:
            start = len(flat_pairs)

            flat_pairs.extend(
                (query, self._document_text(document))
                for document in documents
            )

            offsets.append((start, len(flat_pairs)))

        if not flat_pairs:
            return [[] for _ in query_documents]

        scores = self.model.predict(
            flat_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked_groups: list[list[dict[str, Any]]] = []

        for (_, documents), (start, end) in zip(query_documents, offsets):
            ranked: list[dict[str, Any]] = []

            for document, score in zip(documents, scores[start:end]):
                enriched = dict(document)
                enriched["ann_score"] = enriched["score"]
                enriched["rerank_score"] = float(score)
                ranked.append(enriched)

            ranked.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

            for rank, item in enumerate(ranked, start=1):
                item["rank"] = rank

            ranked_groups.append(ranked)

        return ranked_groups

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        title = str(document.get("title", "")).strip()
        text = str(document.get("text", "")).strip()

        if title and text:
            return f"{title}\n{text}"

        return title or text
