"""Metadata enrichment helpers.

This module keeps metadata extraction separate from retrieval so the pipeline
can later swap the simple deterministic extractor for a LlamaIndex extractor.
"""

from __future__ import annotations

import re

from backonthelangchain.rag.chunking import TextChunk


def _keywords(text: str, *, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "you",
        "your",
        "are",
        "can",
        "from",
        "into",
        "when",
        "then",
        "have",
        "has",
        "should",
        "will",
        "not",
        "faq",
        "question",
        "answer",
    }

    counts: dict[str, int] = {}

    for token in tokens:
        if token not in stopwords:
            counts[token] = counts.get(token, 0) + 1

    return [
        word
        for word, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_keywords]
    ]


def enrich_chunk_metadata(chunks: list[TextChunk]) -> list[TextChunk]:
    """Add simple deterministic metadata to chunks.

    The function name and shape are intentionally compatible with later
    LlamaIndex-style metadata extraction.
    """

    enriched: list[TextChunk] = []

    for chunk in chunks:
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        title = chunk.metadata.get("title") or lines[0].removeprefix("## ")

        summary = ""
        for line in lines:
            if line.lower().startswith("answer:"):
                summary = line.removeprefix("Answer:").strip()
                break

        metadata = {
            **chunk.metadata,
            "title": title,
            "summary": summary[:240],
            "keywords": _keywords(chunk.text),
        }

        enriched.append(
            TextChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                metadata=metadata,
            )
        )

    return enriched
