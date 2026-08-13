"""A tiny, dependency-light vector store backed by numpy.

Chosen over Chroma/FAISS deliberately: on bleeding-edge Python (3.14) native
wheels for those lag, and for a demo corpus a brute-force cosine search over a
normalized matrix is instant and trivial to reason about (which matters for QE).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    score: float
    text: str
    metadata: dict


class VectorStore:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None  # (n, dim), L2-normalized
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, vectors: list[list[float]], texts: list[str], metadatas: list[dict]) -> None:
        if not (len(vectors) == len(texts) == len(metadatas)):
            raise ValueError("vectors, texts and metadatas must be the same length")
        if not vectors:
            return
        arr = _normalize(np.asarray(vectors, dtype=np.float32))
        self._vectors = arr if self._vectors is None else np.vstack([self._vectors, arr])
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def search(self, query_vector: list[float], k: int = 4) -> list[SearchHit]:
        if self._vectors is None or len(self._texts) == 0:
            return []
        q = _normalize(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._vectors @ q  # cosine similarity (both normalized)
        k = min(k, len(scores))
        top = np.argsort(-scores)[:k]
        return [
            SearchHit(float(scores[i]), self._texts[i], self._metadatas[i]) for i in top
        ]

    # --- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), np.float32)
        np.savez(path, vectors=vectors)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"texts": self._texts, "metadatas": self._metadatas})
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        meta_path = path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No index found at {path}. Run ingestion first.")
        data = np.load(path)
        vectors = data["vectors"]
        meta = json.loads(meta_path.read_text())
        store._vectors = vectors if vectors.size else None
        store._texts = meta["texts"]
        store._metadatas = meta["metadatas"]
        return store


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
