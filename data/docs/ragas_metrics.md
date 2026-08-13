# Ragas Evaluation Metrics

Ragas is a framework for evaluating Retrieval-Augmented Generation pipelines.
It provides reference-free and reference-based metrics that use a language model
as a judge to score the quality of retrieval and generation. The four most
widely used metrics are faithfulness, answer relevancy, context precision, and
context recall.

## Faithfulness

Faithfulness measures whether the generated answer is grounded in the retrieved
context. Ragas breaks the answer into individual claims and checks how many of
those claims can be inferred from the retrieved context. The score is the
fraction of claims that are supported. A low faithfulness score is a signal of
hallucination, because the model asserted things the context did not support.

## Answer Relevancy

Answer relevancy measures how directly the answer addresses the user's question.
Ragas generates several questions that the answer could be responding to and
compares them, using embeddings, to the original question. An answer that is
on-topic and complete scores high; an answer that is evasive, padded, or
incomplete scores low.

## Context Precision

Context precision measures whether the relevant chunks are ranked near the top
of the retrieved results. High context precision means the retriever put useful
passages first and did not bury them under irrelevant ones.

## Context Recall

Context recall measures whether all the information needed to answer the
question was actually retrieved. It compares the ground-truth answer against the
retrieved context to check that every required fact is present. Low context
recall points to a retrieval gap rather than a generation problem.
