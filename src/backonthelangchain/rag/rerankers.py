"""Reranker interfaces and Voyage implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import voyageai

from backonthelangchain.rag.retrieval import RetrievedChunk


@dataclass(frozen=True)
class RerankedChunk:
    """A reranked chunk with both retrieval and rerank scores."""

    chunk_id: str
    text: str
    source: str
    metadata: dict
    retrieval_score: float
    rerank_score: float | None


class Reranker(Protocol):
    """Protocol for reranker implementations."""

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]:
        """Return reranked chunks."""


class NoOpReranker:
    """Deterministic fallback that preserves retrieval ordering."""

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]:
        """Return candidates in retriever order."""

        return [
            RerankedChunk(
                chunk_id=item.chunk.chunk_id,
                text=item.chunk.text,
                source=item.chunk.source,
                metadata=item.chunk.metadata,
                retrieval_score=item.score,
                rerank_score=None,
            )
            for item in candidates[:top_k]
        ]


class VoyageReranker:
    """Voyage hosted reranker.

    Requires VOYAGE_API_KEY in the environment.
    """

    def __init__(self, *, model: str = "rerank-2.5"):
        self.model = model
        self.client = voyageai.Client()

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]:
        """Rerank candidate chunks using Voyage."""

        if not candidates:
            return []

        documents = [item.chunk.text for item in candidates]

        result = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=min(top_k, len(documents)),
        )

        reranked: list[RerankedChunk] = []

        for item in result.results:
            candidate = candidates[item.index]

            reranked.append(
                RerankedChunk(
                    chunk_id=candidate.chunk.chunk_id,
                    text=candidate.chunk.text,
                    source=candidate.chunk.source,
                    metadata=candidate.chunk.metadata,
                    retrieval_score=candidate.score,
                    rerank_score=item.relevance_score,
                )
            )

        return reranked
