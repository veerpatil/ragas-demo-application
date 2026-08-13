"""Shared test fixtures.

The unit suite runs fully offline: we force the ``fake`` provider BEFORE any app
module imports ``settings``, and redirect the index/results paths into a tmp dir
so tests never touch a developer's real data.
"""

from __future__ import annotations

import os

# Must be set before app.config is imported anywhere.
os.environ.setdefault("RAG_LLM_PROVIDER", "fake")

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _tmp_paths(tmp_path_factory):
    """Point the store and results dir at a throwaway location for the session."""
    from app.config import settings

    tmp = tmp_path_factory.mktemp("ragqe")
    settings.store_path = tmp / "store.npz"
    settings.results_dir = tmp / "results"
    yield


@pytest.fixture(scope="session")
def store():
    from app.ingest import build_index

    return build_index(persist=False)


@pytest.fixture()
def pipeline(store):
    from app.rag import RagPipeline

    return RagPipeline(store)


@pytest.fixture()
def client():
    """A TestClient with a freshly built (and persisted) index."""
    from fastapi.testclient import TestClient

    from app import api

    resp_store = api.build_index()  # persists to the tmp store_path
    api._pipeline = api.RagPipeline(resp_store)
    return TestClient(api.app)
