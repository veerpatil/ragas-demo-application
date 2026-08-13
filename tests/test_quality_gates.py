"""Quality-gate tests.

Two layers:
  * Offline: the gate/aggregation LOGIC is verified deterministically (no LLM).
  * Live: the real Ragas evaluation runs only when RAGAS_LIVE=1 and Ollama is
    the provider. This is the "does quality clear the bar" regression test.
"""

from __future__ import annotations

import os

import pytest

from eval.runner import _mean, load_history, save_report


def test_mean_ignores_nan():
    assert _mean([1.0, float("nan"), 0.5]) == pytest.approx(0.75)


def test_mean_all_nan_is_nan():
    result = _mean([float("nan")])
    assert result != result  # NaN


def test_save_and_load_history_roundtrip():
    report = {
        "run_id": "20260101T000000Z",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "gen_model": "llama3.2:3b",
        "judge_model": "gemma4:12b",
        "num_samples": 2,
        "metrics": [
            {"name": "faithfulness", "score": 0.9, "threshold": 0.7, "passed": True},
        ],
        "passed": True,
        "per_sample": [{"question": "q", "answer": "a"}],
    }
    save_report(report)
    history = load_history()
    assert history[-1]["run_id"] == "20260101T000000Z"
    # per_sample is stripped from history for brevity.
    assert "per_sample" not in history[-1]


# --- live integration ---------------------------------------------------

RAGAS_LIVE = os.environ.get("RAGAS_LIVE") == "1"


@pytest.mark.skipif(not RAGAS_LIVE, reason="set RAGAS_LIVE=1 (and Ollama running) to run")
def test_live_evaluation_meets_quality_gates():
    from app.config import settings

    assert settings.llm_provider == "ollama", "live eval needs RAG_LLM_PROVIDER=ollama"

    from app.ingest import build_index
    from app.rag import RagPipeline
    from eval.runner import run_evaluation

    pipeline = RagPipeline(build_index(persist=False))
    report = run_evaluation(sample_size=2, pipeline=pipeline, persist=False)

    failed = [m["name"] for m in report["metrics"] if not m["passed"]]
    assert not failed, f"metrics below gate: {failed}"
