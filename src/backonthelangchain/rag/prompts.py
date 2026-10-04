"""Prompt helpers for RAG context injection."""

from __future__ import annotations

from pathlib import Path

from backonthelangchain.rag.rerankers import RerankedChunk


def format_rag_context(chunks: list[RerankedChunk]) -> str:
    """Format reranked chunks as compact support context."""

    if not chunks:
        return "No relevant FAQ context was found."

    blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title", f"FAQ {index}")
        source = Path(chunk.metadata.get("source", chunk.source)).name

        blocks.append(
            "\n".join(
                [
                    f"[FAQ {index}] {title}",
                    f"Source: {source}",
                    f"Chunk ID: {chunk.chunk_id}",
                    chunk.text,
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)
