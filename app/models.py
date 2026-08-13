"""Model access layer: embeddings + chat generation.

Two providers:
  * "ollama" -> real local models via the ``ollama`` python client.
  * "fake"   -> deterministic, network-free stubs for offline tests/CI.

The fake embedder is a hashing bag-of-words vectorizer, so it still produces
meaningful cosine similarity (chunks sharing vocabulary with the query rank
higher). The fake chat model answers by extracting the sentence from the
provided context most relevant to the question, which keeps the RAG contract
testable without an LLM.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from app.config import settings

_FAKE_DIM = 256
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _fake_embed_one(text: str) -> list[float]:
    vec = [0.0] * _FAKE_DIM
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _FAKE_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@lru_cache(maxsize=1)
def _ollama_client():
    import ollama

    return ollama.Client(host=settings.ollama_base_url)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into vectors."""
    if settings.llm_provider == "fake":
        return [_fake_embed_one(t) for t in texts]
    client = _ollama_client()
    out: list[list[float]] = []
    for t in texts:
        resp = client.embeddings(model=settings.embed_model, prompt=t)
        out.append(list(resp["embedding"]))
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def _fake_answer(question: str, contexts: list[str]) -> str:
    """Pick the context sentence with the most word overlap with the question."""
    q = set(_tokenize(question))
    sentences: list[str] = []
    for c in contexts:
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", c) if s.strip())
    if not sentences:
        return "I don't have enough information to answer that."
    best = max(sentences, key=lambda s: len(q & set(_tokenize(s))))
    return best


def generate(question: str, contexts: list[str]) -> str:
    """Generate a grounded answer from the retrieved contexts."""
    if settings.llm_provider == "fake":
        return _fake_answer(question, contexts)

    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    system = (
        "You are a precise assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain the answer, say you "
        "don't know. Be concise."
    )
    user = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    client = _ollama_client()
    resp = client.chat(
        model=settings.gen_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.0},
    )
    return resp["message"]["content"].strip()
