"""FastAPI entry point for the Faiss retrieval service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.retriever import RetrievalService

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts/fiqa_ivfpq_m96")
retriever = RetrievalService(ARTIFACT_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    retriever.load()
    yield


app = FastAPI(
    title="RAG Embedding Compression Retrieval API",
    version="0.2.0",
    description=(
        "A Faiss IVF-PQ retrieval service backed by the FiQA benchmark artifacts. "
        "The batch endpoint encodes queries and searches the index in one Faiss call."
    ),
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["What is a dividend stock?"])
    top_k: int = Field(default=5, ge=1, le=20)
    nprobe: int | None = Field(default=None, ge=1, le=256)


class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=32)
    top_k: int = Field(default=5, ge=1, le=20)
    nprobe: int | None = Field(default=None, ge=1, le=256)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if retriever.is_ready else "starting",
        "artifact_dir": ARTIFACT_DIR,
        "index_type": retriever.config.get("index_type"),
        "embedding_model": retriever.config.get("embedding_model"),
        "document_count": len(retriever.documents),
    }


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    try:
        return retriever.search(
            query=payload.query,
            top_k=payload.top_k,
            nprobe=payload.nprobe,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch-search")
def batch_search(payload: BatchSearchRequest) -> dict:
    try:
        return retriever.search_many(
            queries=payload.queries,
            top_k=payload.top_k,
            nprobe=payload.nprobe,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
