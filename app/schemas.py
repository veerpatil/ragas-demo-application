"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is faithfulness in Ragas?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceRef(BaseModel):
    source: str | None
    chunk: int | None
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    sources: list[SourceRef]


class IngestResponse(BaseModel):
    chunks_indexed: int
    store_path: str


class EvaluateRequest(BaseModel):
    sample_size: int | None = Field(default=None, ge=1, description="Limit golden items evaluated")


class MetricSummary(BaseModel):
    name: str
    score: float
    threshold: float
    passed: bool


class EvaluateResponse(BaseModel):
    run_id: str
    timestamp: str
    gen_model: str
    judge_model: str
    num_samples: int
    metrics: list[MetricSummary]
    passed: bool


class HistoryResponse(BaseModel):
    runs: list[EvaluateResponse]
