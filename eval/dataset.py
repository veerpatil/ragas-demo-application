"""The golden dataset: questions with known-good reference answers.

This is the fixed yardstick every pipeline change is measured against. Each
reference answer is derivable from the bundled corpus in ``data/docs``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenItem:
    question: str
    reference: str


GOLDEN_DATASET: list[GoldenItem] = [
    GoldenItem(
        question="What does faithfulness measure in Ragas?",
        reference=(
            "Faithfulness measures whether the generated answer is grounded in the "
            "retrieved context. It is the fraction of the answer's claims that can be "
            "inferred from the context, and a low score signals hallucination."
        ),
    ),
    GoldenItem(
        question="What are the two core stages of a RAG pipeline?",
        reference=(
            "A RAG pipeline has a retrieval stage, which embeds the question and finds "
            "the most relevant document chunks, and a generation stage, which inserts "
            "those chunks into the prompt so the model produces a grounded answer."
        ),
    ),
    GoldenItem(
        question="What is cosine similarity and what does a score of 1.0 mean?",
        reference=(
            "Cosine similarity compares two embeddings by the angle between them rather "
            "than their magnitude. A score of 1.0 means the vectors point in exactly the "
            "same direction; 0.0 means they are unrelated."
        ),
    ),
    GoldenItem(
        question="What is a quality gate in the context of RAG evaluation?",
        reference=(
            "A quality gate is a threshold on an evaluation metric that a build must clear "
            "to be acceptable, for example requiring mean faithfulness above 0.70. If a "
            "change drops a metric below its gate, the build fails."
        ),
    ),
    GoldenItem(
        question="Why does chunk size matter when indexing documents?",
        reference=(
            "Chunk size affects retrieval precision: chunks that are too large dilute the "
            "embedding, while chunks that are too small lose surrounding context. A "
            "moderate size with a small overlap keeps each unit focused without losing "
            "ideas that span a boundary."
        ),
    ),
    GoldenItem(
        question="What does context recall measure and what does a low score indicate?",
        reference=(
            "Context recall measures whether all the information needed to answer the "
            "question was retrieved, by comparing the ground-truth answer against the "
            "retrieved context. A low score points to a retrieval gap rather than a "
            "generation problem."
        ),
    ),
    GoldenItem(
        question="Why must the same embedding model be used for documents and queries?",
        reference=(
            "The same embedding model must be used for both indexing and querying because "
            "otherwise the document and query vectors live in different spaces and are not "
            "comparable."
        ),
    ),
    GoldenItem(
        question="Why does the judge model choice matter in Ragas evaluation?",
        reference=(
            "The judge model must be capable enough to follow structured-output "
            "instructions reliably. A judge that is too small may return malformed output "
            "and produce noisy scores, so the judge model is itself a quality decision."
        ),
    ),
]
