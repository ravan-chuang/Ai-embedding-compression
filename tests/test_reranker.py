from __future__ import annotations

import numpy as np
import pytest

from app.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert batch_size == 16
        assert show_progress_bar is False
        assert pairs[0] == ("query", "Alpha\nfirst document")
        return np.array([0.2, 0.9], dtype=np.float32)


def test_reranker_sorts_candidates_and_keeps_ann_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.reranker.CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker()
    reranker.load()

    ranked = reranker.rerank(
        "query",
        [
            {
                "rank": 1,
                "score": 0.8,
                "title": "Alpha",
                "text": "first document",
            },
            {
                "rank": 2,
                "score": 0.7,
                "title": "",
                "text": "second document",
            },
        ],
    )

    assert [item["rerank_score"] for item in ranked] == pytest.approx([0.9, 0.2])
    assert [item["ann_score"] for item in ranked] == pytest.approx([0.7, 0.8])
    assert [item["rank"] for item in ranked] == [1, 2]
