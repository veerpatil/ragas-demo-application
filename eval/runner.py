"""Runs the golden dataset through the RAG pipeline, scores it with Ragas,
applies the quality gates, and persists each run so quality can be tracked
over time.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.rag import RagPipeline
from eval.dataset import GOLDEN_DATASET, GoldenItem


def _metric_value(result) -> float:
    return float(getattr(result, "value", result))


async def _score_sample(metrics, sample: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, metric, build_kwargs in metrics:
        try:
            result = await metric.ascore(**build_kwargs(sample))
            scores[name] = _metric_value(result)
        except Exception as exc:  # a single flaky metric shouldn't kill the run
            print(f"  ! {name} failed on this sample: {exc!r}")
            scores[name] = float("nan")
    return scores


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return sum(clean) / len(clean) if clean else float("nan")


async def _run_async(items: list[GoldenItem], pipeline: RagPipeline) -> dict:
    from eval.judge import build_metrics

    metrics = build_metrics()
    metric_names = [name for name, _, _ in metrics]

    per_sample: list[dict] = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.question}")
        result = pipeline.query(item.question)
        sample = {
            "question": item.question,
            "answer": result.answer,
            "contexts": result.contexts,
            "reference": item.reference,
        }
        scores = await _score_sample(metrics, sample)
        per_sample.append({**sample, "scores": scores})
        print("     " + "  ".join(f"{n}={scores[n]:.2f}" for n in metric_names))

    aggregate = {
        name: _mean([s["scores"][name] for s in per_sample]) for name in metric_names
    }
    gates = settings.gates
    metric_summaries = [
        {
            "name": name,
            "score": round(aggregate[name], 4),
            "threshold": gates.get(name, 0.0),
            "passed": bool(aggregate[name] >= gates.get(name, 0.0)),
        }
        for name in metric_names
    ]
    passed = all(m["passed"] for m in metric_summaries)

    now = datetime.now(timezone.utc)
    return {
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(),
        "gen_model": settings.gen_model,
        "judge_model": settings.judge_model,
        "num_samples": len(items),
        "metrics": metric_summaries,
        "passed": passed,
        "per_sample": per_sample,
    }


def run_evaluation(
    sample_size: int | None = None,
    pipeline: RagPipeline | None = None,
    persist: bool = True,
) -> dict:
    """Run the full evaluation. Requires a live Ollama judge."""
    items = GOLDEN_DATASET[:sample_size] if sample_size else GOLDEN_DATASET
    pipeline = pipeline or RagPipeline()
    report = asyncio.run(_run_async(items, pipeline))
    if persist:
        save_report(report)
    return report


def save_report(report: dict) -> Path:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def load_history() -> list[dict]:
    """Return every persisted run, oldest first (per_sample stripped for brevity)."""
    if not settings.results_dir.exists():
        return []
    runs: list[dict] = []
    for path in sorted(settings.results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        runs.append({k: v for k, v in data.items() if k != "per_sample"})
    return runs


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    report = run_evaluation(sample_size=n)
    print("\n=== Summary ===")
    for m in report["metrics"]:
        flag = "PASS" if m["passed"] else "FAIL"
        print(f"  {m['name']:<20} {m['score']:.3f}  (gate {m['threshold']:.2f})  [{flag}]")
    print(f"\nOverall: {'PASS' if report['passed'] else 'FAIL'}")
    sys.exit(0 if report["passed"] else 1)
