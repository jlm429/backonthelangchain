"""Embedding helpers for RAG.

Uses OpenAI embeddings so the project can reuse the existing OPENAI_API_KEY.
"""

from __future__ import annotations

from openai import OpenAI


class OpenAIEmbeddingModel:
    """Small wrapper around OpenAI's embeddings API."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ):
        self.model = model
        self.client = client or OpenAI()

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""

        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]
