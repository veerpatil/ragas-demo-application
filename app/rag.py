"""The RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.models import embed_query, generate
from app.vectorstore import VectorStore


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str]
    sources: list[dict] = field(default_factory=list)


class RagPipeline:
    def __init__(self, store: VectorStore | None = None) -> None:
        self._store = store or VectorStore.load(settings.store_path)

    @property
    def store(self) -> VectorStore:
        return self._store

    def retrieve(self, question: str, top_k: int | None = None):
        return self._store.search(embed_query(question), k=top_k or settings.top_k)

    def query(self, question: str, top_k: int | None = None) -> RagResult:
        hits = self.retrieve(question, top_k)
        contexts = [h.text for h in hits]
        answer = generate(question, contexts)
        sources = [
            {"source": h.metadata.get("source"), "chunk": h.metadata.get("chunk"), "score": round(h.score, 4)}
            for h in hits
        ]
        return RagResult(question=question, answer=answer, contexts=contexts, sources=sources)
