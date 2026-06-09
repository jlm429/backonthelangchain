"""Tech support service backed by a RAG pipeline."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backonthelangchain.agents.prompts import TECH_SUPPORT_PROMPT
from backonthelangchain.rag.pipelines import TechSupportRAGPipeline


class TechSupportRAGService:
    """Answer tech support questions using retrieved FAQ context."""

    def __init__(
        self,
        chat_model,
        rag_pipeline: TechSupportRAGPipeline,
    ):
        self.chat_model = chat_model
        self.rag_pipeline = rag_pipeline

    def answer(self, user_query: str) -> dict:
        """Retrieve FAQ context and generate a support answer."""

        rag_result = self.rag_pipeline.run(user_query)

        response = self.chat_model.invoke(
            [
                TECH_SUPPORT_PROMPT,
                SystemMessage(
                    content=(
                        "Relevant Tier 1 FAQ context:\n\n"
                        f"{rag_result.context}\n\n"
                        "Use the FAQ context whenever it is relevant.\n\n"
                        "If one or more FAQ entries support your answer, include a "
                        "'Relevant FAQ' section at the end listing the FAQ title(s) used.\n\n"
                        "If the FAQ context does not fully answer the question, say so briefly "
                        "and provide the best available guidance.\n\n"
                        "Do not invent FAQ content that was not provided in the context."
                    )
                ),
                HumanMessage(content=user_query),
            ]
        )

        return {
            "answer": response.content,
            "rag_context": rag_result.context,
            "rag_sources": [
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.metadata.get("title"),
                    "source": chunk.source,
                    "retrieval_score": chunk.retrieval_score,
                    "rerank_score": chunk.rerank_score,
                }
                for chunk in rag_result.reranked_chunks
            ],
        }