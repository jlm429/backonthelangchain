"""End-to-end deterministic RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backonthelangchain.rag.chunking import split_markdown_faqs
from backonthelangchain.rag.embeddings import OpenAIEmbeddingModel
from backonthelangchain.rag.loaders import load_text_file
from backonthelangchain.rag.metadata import enrich_chunk_metadata
from backonthelangchain.rag.prompts import format_rag_context
from backonthelangchain.rag.rerankers import NoOpReranker, RerankedChunk, Reranker
from backonthelangchain.rag.retrieval import FAISSRetriever, RetrievedChunk


DEFAULT_FAQ_PATH = Path(__file__).parent / "data" / "tier1_tech_support_faq.md"


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

        FAQ markdown
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
            raw_document = load_text_file(self.faq_path)
            chunks = split_markdown_faqs(raw_document)
            enriched_chunks = enrich_chunk_metadata(chunks)

            self.retriever = FAISSRetriever.from_chunks(
                enriched_chunks,
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
