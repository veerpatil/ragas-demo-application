# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a technique that combines a retrieval system
with a generative language model. Instead of relying only on the parameters
learned during training, a RAG system fetches relevant documents from an
external knowledge base at query time and passes them to the language model as
context. This grounds the generated answer in real source material and reduces
hallucination.

## The two stages

A RAG pipeline has two core stages. The first stage is retrieval: the user
question is embedded into a vector and compared against a vector index of
document chunks to find the most relevant passages. The second stage is
generation: the retrieved passages are inserted into the prompt and the language
model produces an answer grounded in that context.

## Why chunking matters

Documents are split into smaller chunks before they are embedded. Chunking
keeps each embedded unit focused on a single topic, which improves retrieval
precision. If chunks are too large they dilute the embedding; if they are too
small they lose surrounding context. A common strategy uses a moderate chunk
size with a small overlap between adjacent chunks so that ideas spanning a
boundary are not lost.

## Grounding and citations

Because the answer is generated from retrieved passages, a RAG system can cite
its sources. Returning the source documents alongside the answer lets users
verify claims and builds trust in the system.
