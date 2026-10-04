"""End-to-end deterministic RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backonthelangchain.rag.chunking import TextChunk, split_markdown_faqs
from backonthelangchain.rag.embeddings import OpenAIEmbeddingModel
from backonthelangchain.rag.loaders import load_text_file
from backonthelangchain.rag.metadata import enrich_chunk_metadata
from backonthelangchain.rag.prompts import format_rag_context
from backonthelangchain.rag.rerankers import NoOpReranker, RerankedChunk, Reranker
from backonthelangchain.rag.retrieval import FAISSRetriever, RetrievedChunk


DEFAULT_FAQ_PATH = Path(__file__).parent / "data" / "demo_support_knowledge"


def load_support_knowledge_chunks(path: str | Path) -> list[TextChunk]:
    """Load one Markdown file or a directory of demo support documents."""
    knowledge_path = Path(path)
    document_paths = (
        sorted(knowledge_path.glob("*.md"))
        if knowledge_path.is_dir()
        else [knowledge_path]
    )
    chunks = []
    for document_path in document_paths:
        raw_document = load_text_file(document_path)
        chunks.extend(enrich_chunk_metadata(split_markdown_faqs(raw_document)))
    if not chunks:
        raise ValueError("Support knowledge base must contain at least one FAQ section.")
    return chunks


@dataclass(frozen=True)
class RAGPipelineResult:
    """Output of the support RAG retrieval pipeline."""

    query: str
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RerankedChunk]
    context: str


class TechSupportRAGPipeline:
    """Retrieve FAQ context for Tier 1 technical support questions.

    Pipeline:

        fictional demo support Markdown documents
        -> FAQ boundary chunking
        -> deterministic metadata enrichment
        -> OpenAI embeddings
        -> FAISS top-10 vector retrieval
        -> Voyage rerank-2.5 top-5
        -> formatted context for the support prompt
    """

    def __init__(
        self,
        *,
        faq_path: str | Path = DEFAULT_FAQ_PATH,
        embedding_model: OpenAIEmbeddingModel | None = None,
        retriever: FAISSRetriever | None = None,
        reranker: Reranker | None = None,
        retrieve_top_k: int = 10,
        rerank_top_k: int = 5,
    ):
        self.faq_path = Path(faq_path)
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.reranker = reranker or NoOpReranker()

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = FAISSRetriever.from_chunks(
                load_support_knowledge_chunks(self.faq_path),
                embedding_model=self.embedding_model,
            )

    @classmethod
    def from_saved_index(
        cls,
        *,
        index_directory: str | Path,
        faq_path: str | Path = DEFAULT_FAQ_PATH,
        embedding_model: OpenAIEmbeddingModel | None = None,
        reranker: Reranker | None = None,
        retrieve_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> "TechSupportRAGPipeline":
        """Create the pipeline from a persisted FAISS index."""

        embedding_model = embedding_model or OpenAIEmbeddingModel()

        retriever = FAISSRetriever.load(
            index_directory,
            embedding_model=embedding_model,
        )

        return cls(
            faq_path=faq_path,
            embedding_model=embedding_model,
            retriever=retriever,
            reranker=reranker,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
        )

    def save_index(self, directory: str | Path) -> None:
        """Persist the current FAISS index and chunks."""

        self.retriever.save(directory)

    def run(self, query: str) -> RAGPipelineResult:
        """Retrieve, rerank, and format support FAQ context."""

        retrieved = self.retriever.retrieve(
            query,
            top_k=self.retrieve_top_k,
        )

        reranked = self.reranker.rerank(
            query=query,
            candidates=retrieved,
            top_k=self.rerank_top_k,
        )

        return RAGPipelineResult(
            query=query,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            context=format_rag_context(reranked),
        )
