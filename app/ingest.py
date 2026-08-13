"""Ingestion: load the document corpus, chunk it, embed it, persist the index."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.models import embed_texts
from app.vectorstore import VectorStore

_SUFFIXES = {".md", ".txt"}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries, packing paragraphs up to ~chunk_size chars
    with a sliding character overlap between adjacent chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > chunk_size:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n\n" + para) if overlap else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Return (source_name, text) for every supported file in the corpus."""
    docs: list[tuple[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in _SUFFIXES:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_index(
    data_dir: Path | None = None,
    store_path: Path | None = None,
    persist: bool = True,
) -> VectorStore:
    data_dir = data_dir or settings.data_dir
    store_path = store_path or settings.store_path

    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No .md/.txt documents found in {data_dir}")

    texts: list[str] = []
    metadatas: list[dict] = []
    for source, content in documents:
        for i, chunk in enumerate(chunk_text(content, settings.chunk_size, settings.chunk_overlap)):
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    vectors = embed_texts(texts)
    store = VectorStore()
    store.add(vectors, texts, metadatas)
    if persist:
        store.save(store_path)
    return store


if __name__ == "__main__":
    store = build_index()
    print(f"Indexed {len(store)} chunks -> {settings.store_path}")
