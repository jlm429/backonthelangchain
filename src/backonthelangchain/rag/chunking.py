"""Chunking helpers for deterministic local RAG examples."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from backonthelangchain.rag.loaders import RawDocument


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text plus lightweight metadata."""

    chunk_id: str
    text: str
    source: str
    metadata: dict


def _stable_chunk_id(source: str, index: int, text: str) -> str:
    digest = sha256(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def split_markdown_faqs(document: RawDocument) -> list[TextChunk]:
    """Split a markdown FAQ file into one chunk per FAQ section.

    Expected format:

        ## FAQ title
        Question: ...
        Answer: ...

    This intentionally uses FAQ boundaries rather than blind character windows.
    """

    sections: list[str] = []
    current: list[str] = []

    for line in document.text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    chunks: list[TextChunk] = []

    for index, section in enumerate(sections):
        if not section or section.startswith("# "):
            continue

        title = section.splitlines()[0].removeprefix("## ").strip()
        chunk_id = _stable_chunk_id(document.source, index, section)

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=section,
                source=document.source,
                metadata={
                    "source": document.source,
                    "title": title,
                    "chunk_index": index,
                    "chunk_type": "faq",
                },
            )
        )

    return chunks
