"""Unit tests for the numpy vector store."""

from __future__ import annotations

import pytest

from app.vectorstore import VectorStore


def test_add_and_search_ranks_by_similarity():
    store = VectorStore()
    store.add(
        vectors=[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        texts=["east", "north", "mostly east"],
        metadatas=[{"i": 0}, {"i": 1}, {"i": 2}],
    )
    hits = store.search([1.0, 0.0], k=2)
    assert [h.text for h in hits] == ["east", "mostly east"]
    assert hits[0].score >= hits[1].score


def test_search_on_empty_store_returns_empty():
    assert VectorStore().search([1.0, 0.0], k=3) == []


def test_add_length_mismatch_raises():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.add(vectors=[[1.0, 0.0]], texts=["a", "b"], metadatas=[{}])


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add([[1.0, 2.0], [3.0, 4.0]], ["a", "b"], [{"s": "x"}, {"s": "y"}])
    path = tmp_path / "store.npz"
    store.save(path)

    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    hit = loaded.search([1.0, 2.0], k=1)[0]
    assert hit.text == "a"
    assert hit.metadata["s"] == "x"


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope.npz")
