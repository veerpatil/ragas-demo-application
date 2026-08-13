# Vector Search and Embeddings

Vector search is the retrieval mechanism at the heart of most RAG systems. Text
is converted into a high-dimensional numeric vector called an embedding. Texts
with similar meaning map to vectors that are close together, so semantic search
becomes a matter of finding the nearest vectors to the query vector.

## Cosine similarity

The most common way to compare two embeddings is cosine similarity, which
measures the angle between the vectors rather than their magnitude. A cosine
similarity of 1.0 means the vectors point in exactly the same direction, while
0.0 means they are unrelated. Normalizing vectors to unit length before
comparison makes cosine similarity equivalent to a simple dot product.

## Embedding models

An embedding model turns text into vectors. Smaller models such as all-MiniLM
produce compact 384-dimensional vectors and are fast, while models such as
nomic-embed-text produce 768-dimensional vectors that capture more nuance. The
same embedding model must be used for both indexing documents and embedding
queries, otherwise the vectors are not comparable.

## Top-k retrieval

At query time the system embeds the question and returns the top-k most similar
chunks. Choosing k is a trade-off: a larger k improves recall by including more
candidate passages, but it also adds noise and consumes more of the language
model's context window.
