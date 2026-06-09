"""FAISS vector retrieval using OpenAI embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import faiss
import numpy as np

from backonthelangchain.rag.chunking import TextChunk
from backonthelangchain.rag.embeddings import OpenAIEmbeddingModel


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk and its vector similarity score."""

    chunk: TextChunk
    score: float


class FAISSRetriever:
    """In-memory FAISS retriever for local RAG examples.

    The retriever embeds chunks once at construction time unless a saved FAISS
    index is loaded. For larger corpora, build and persist the index offline.
    """

    def __init__(
        self,
        *,
        chunks: list[TextChunk],
        embedding_model: OpenAIEmbeddingModel | None = None,
    ):
        if not chunks:
            raise ValueError("FAISSRetriever requires at least one chunk.")

        self.chunks = chunks
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.index: faiss.IndexFlatIP | None = None

    @classmethod
    def from_chunks(
        cls,
        chunks: list[TextChunk],
        *,
        embedding_model: OpenAIEmbeddingModel | None = None,
    ) -> "FAISSRetriever":
        """Build a FAISS retriever from chunks."""

        retriever = cls(
            chunks=chunks,
            embedding_model=embedding_model,
        )
        retriever.build_index()
        return retriever

    def build_index(self) -> None:
        """Embed chunks and build an inner-product FAISS index.

        Vectors are L2-normalized, so inner product is cosine similarity.
        """

        texts = [chunk.text for chunk in self.chunks]
        vectors = np.array(
            self.embedding_model.embed_documents(texts),
            dtype="float32",
        )

        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index = index

    def retrieve(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        """Return top-k chunks by vector similarity."""

        if self.index is None:
            self.build_index()

        query_vector = np.array(
            [self.embedding_model.embed_query(query)],
            dtype="float32",
        )
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.chunks)),
        )

        results: list[RetrievedChunk] = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(index)],
                    score=float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and chunk metadata."""

        if self.index is None:
            self.build_index()

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with (path / "chunks.pkl").open("wb") as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        embedding_model: OpenAIEmbeddingModel | None = None,
    ) -> "FAISSRetriever":
        """Load a persisted FAISS retriever."""

        path = Path(directory)

        with (path / "chunks.pkl").open("rb") as file:
            chunks = pickle.load(file)

        retriever = cls(
            chunks=chunks,
            embedding_model=embedding_model,
        )
        retriever.index = faiss.read_index(str(path / "index.faiss"))

        return retriever
