"""FastAPI REST service for the RAG + QE platform.

Endpoints
    GET  /health            liveness + index/provider status
    POST /ingest            (re)build the vector index from the corpus
    POST /query             ask a question, get a grounded answer + sources
    POST /evaluate          run the Ragas golden-dataset eval (needs Ollama judge)
    GET  /metrics/history   every past eval run (for the dashboard)
    GET  /metrics/latest    the most recent eval run
    GET  /                  the metrics dashboard (static HTML)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.ingest import build_index
from app.rag import RagPipeline
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HistoryResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.vectorstore import VectorStore
from eval.runner import load_history, run_evaluation

app = FastAPI(title="RAG QE Platform", version="0.1.0")

_DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"

# Cached pipeline; rebuilt on ingest.
_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RagPipeline(VectorStore.load(settings.store_path))
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="No index found. POST /ingest first to build the vector store.",
            )
    return _pipeline


@app.get("/health")
def health() -> dict:
    indexed = settings.store_path.exists()
    return {"status": "ok", "provider": settings.llm_provider, "indexed": indexed}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    global _pipeline
    store = build_index()
    _pipeline = RagPipeline(store)
    return IngestResponse(chunks_indexed=len(store), store_path=str(settings.store_path))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = get_pipeline().query(req.question, top_k=req.top_k)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        contexts=result.contexts,
        sources=result.sources,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    # Sync endpoint on purpose: FastAPI runs it in a worker thread, so the
    # runner's internal asyncio.run() gets a clean event loop.
    if settings.llm_provider != "ollama":
        raise HTTPException(
            status_code=400,
            detail="Evaluation needs a live Ollama judge; set RAG_LLM_PROVIDER=ollama.",
        )
    report = run_evaluation(sample_size=req.sample_size, pipeline=get_pipeline())
    return EvaluateResponse(**{k: v for k, v in report.items() if k != "per_sample"})


@app.get("/metrics/history", response_model=HistoryResponse)
def metrics_history() -> HistoryResponse:
    return HistoryResponse(runs=[EvaluateResponse(**r) for r in load_history()])


@app.get("/metrics/latest", response_model=EvaluateResponse)
def metrics_latest() -> EvaluateResponse:
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No evaluation runs yet.")
    return EvaluateResponse(**history[-1])


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_DASHBOARD)
