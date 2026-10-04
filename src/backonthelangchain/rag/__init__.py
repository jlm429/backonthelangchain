"""RAG utilities and pipelines for backonthelangchain."""

from typing import Any

__all__ = [
    "TechSupportRAGPipeline",
    "VoyageReranker",
]


def __getattr__(name: str) -> Any:
    """Load optional RAG dependencies only when their public types are used."""
    if name == "TechSupportRAGPipeline":
        from backonthelangchain.rag.pipelines import TechSupportRAGPipeline

        return TechSupportRAGPipeline
    if name == "VoyageReranker":
        from backonthelangchain.rag.rerankers import VoyageReranker

        return VoyageReranker
    raise AttributeError(name)
