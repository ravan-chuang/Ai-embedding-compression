from __future__ import annotations

import numpy as np
import pytest

from app.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    calls: list[list[tuple[str, str]]] = []

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

        type(self).calls.append(pairs)

        scores = {
            ("query-a", "Alpha\nfirst document"): 0.2,
            ("query-a", "second document"): 0.9,
            ("query-b", "third document"): 0.4,
            ("query-b", "Beta\nfourth document"): 0.8,
        }

        return np.array(
            [scores[pair] for pair in pairs],
            dtype=np.float32,
        )


def test_reranker_sorts_candidates_and_keeps_ann_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.reranker.CrossEncoder", FakeCrossEncoder)
    FakeCrossEncoder.calls = []

    reranker = CrossEncoderReranker()
    reranker.load()

    ranked = reranker.rerank(
        "query-a",
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

    assert len(FakeCrossEncoder.calls) == 1
    assert [item["rerank_score"] for item in ranked] == pytest.approx(
        [0.9, 0.2]
    )
    assert [item["ann_score"] for item in ranked] == pytest.approx(
        [0.7, 0.8]
    )
    assert [item["rank"] for item in ranked] == [1, 2]


def test_rerank_many_uses_one_prediction_call_for_multiple_queries(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.reranker.CrossEncoder", FakeCrossEncoder)
    FakeCrossEncoder.calls = []

    reranker = CrossEncoderReranker()
    reranker.load()

    ranked_groups = reranker.rerank_many(
        [
            (
                "query-a",
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
            ),
            (
                "query-b",
                [
                    {
                        "rank": 1,
                        "score": 0.6,
                        "title": "",
                        "text": "third document",
                    },
                    {
                        "rank": 2,
                        "score": 0.5,
                        "title": "Beta",
                        "text": "fourth document",
                    },
                ],
            ),
        ]
    )

    assert len(FakeCrossEncoder.calls) == 1
    assert len(FakeCrossEncoder.calls[0]) == 4

    assert [item["rerank_score"] for item in ranked_groups[0]] == pytest.approx(
        [0.9, 0.2]
    )
    assert [item["rerank_score"] for item in ranked_groups[1]] == pytest.approx(
        [0.8, 0.4]
    )
    assert [item["rank"] for item in ranked_groups[0]] == [1, 2]
    assert [item["rank"] for item in ranked_groups[1]] == [1, 2]
