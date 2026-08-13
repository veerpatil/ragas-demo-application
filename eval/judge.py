"""Builds the Ragas metric objects backed by a local Ollama judge.

Ollama exposes an OpenAI-compatible endpoint, which is exactly what the modern
Ragas ``collections`` metrics expect (an Instructor LLM built from an
``AsyncOpenAI`` client). Only the LLM-judge metrics are constructed here; the
embedding-based metric additionally needs an embedding model.
"""

from __future__ import annotations

from app.config import settings


def build_judge():
    """Return (llm, embeddings) Ragas objects driven by the local Ollama judge."""
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    client = AsyncOpenAI(base_url=settings.openai_base_url, api_key="ollama")
    llm = llm_factory(settings.judge_model, client=client)
    # Ollama speaks the OpenAI protocol, so the provider is "openai".
    embeddings = embedding_factory(
        "openai", settings.embed_model, client=client, interface="modern"
    )
    return llm, embeddings


def build_metrics():
    """Instantiate the four core Ragas metrics.

    Returns a list of (name, metric, kwargs_builder) where kwargs_builder maps a
    sample dict to the keyword arguments that metric's ``ascore`` expects.
    """
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    llm, embeddings = build_judge()

    return [
        (
            "faithfulness",
            Faithfulness(llm=llm),
            lambda s: dict(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
            ),
        ),
        (
            "answer_relevancy",
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            lambda s: dict(user_input=s["question"], response=s["answer"]),
        ),
        (
            "context_precision",
            ContextPrecisionWithReference(llm=llm),
            lambda s: dict(
                user_input=s["question"],
                reference=s["reference"],
                retrieved_contexts=s["contexts"],
            ),
        ),
        (
            "context_recall",
            ContextRecall(llm=llm),
            lambda s: dict(
                user_input=s["question"],
                retrieved_contexts=s["contexts"],
                reference=s["reference"],
            ),
        ),
    ]
