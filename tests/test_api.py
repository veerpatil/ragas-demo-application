"""API contract tests (offline, fake provider)."""

from __future__ import annotations


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "fake"


def test_query_returns_answer_and_sources(client):
    resp = client.post("/query", json={"question": "What is cosine similarity?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert len(body["contexts"]) == 3
    assert len(body["sources"]) == 3
    assert all("score" in s for s in body["sources"])


def test_query_validation_rejects_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_ingest_reports_chunk_count(client):
    body = client.post("/ingest").json()
    assert body["chunks_indexed"] > 0


def test_evaluate_blocked_without_ollama(client):
    # In fake mode the eval endpoint must refuse rather than produce fake scores.
    resp = client.post("/evaluate", json={})
    assert resp.status_code == 400


def test_metrics_history_empty_initially(client, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "results_dir", tmp_path / "empty")
    body = client.get("/metrics/history").json()
    assert body["runs"] == []


def test_metrics_latest_404_when_no_runs(client, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "results_dir", tmp_path / "empty")
    assert client.get("/metrics/latest").status_code == 404
