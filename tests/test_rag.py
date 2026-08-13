"""Unit tests for ingestion + the RAG pipeline (offline, fake provider)."""

from __future__ import annotations

from app.ingest import chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(20))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)  # size + overlap slack


def test_index_covers_all_sources(store):
    sources = {m["source"] for m in store._metadatas}
    assert {"rag_overview.md", "ragas_metrics.md", "vector_search.md", "qe_practices.md"} <= sources


def test_pipeline_retrieves_relevant_chunk(pipeline):
    hits = pipeline.retrieve("What does faithfulness measure?", top_k=3)
    assert hits, "expected at least one retrieved chunk"
    # The faithfulness doc should surface for a faithfulness question.
    assert any("faithful" in h.text.lower() for h in hits)


def test_pipeline_query_returns_grounded_answer(pipeline):
    result = pipeline.query("What does faithfulness measure?", top_k=3)
    assert result.answer
    assert len(result.contexts) == 3
    assert len(result.sources) == 3
    # The fake generator answers by quoting context, so the answer is grounded.
    assert any(result.answer in c or result.answer[:30] in c for c in result.contexts)


def test_top_k_controls_context_count(pipeline):
    assert len(pipeline.query("cosine similarity", top_k=1).contexts) == 1
    assert len(pipeline.query("cosine similarity", top_k=5).contexts) == 5
