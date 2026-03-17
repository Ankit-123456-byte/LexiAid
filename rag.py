"""
rag.py — Pinecone RAG for jurisdiction-specific legal context.

Responsibilities:
  - Embed and store Indian legal reference documents in Pinecone
  - Retrieve the most relevant legal context chunks for a given document
  - Gracefully degrade (return empty string) if Pinecone is not configured
"""

import os
from typing import Optional

INDEX_NAME    = "lexiaid-legal-docs"
EMBED_DIM     = 384   # all-MiniLM-L6-v2 output dimension
CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_pc     = None
_index  = None
_embedder = None


def _get_pinecone():
    global _pc
    if _pc is None:
        api_key = os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set in environment")
        from pinecone import Pinecone
        _pc = Pinecone(api_key=api_key)
    return _pc


def _get_index():
    global _index
    if _index is None:
        from pinecone import ServerlessSpec
        pc = _get_pinecone()
        existing = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index(INDEX_NAME)
    return _index


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _chunk_text(text: str) -> list:
    """Sliding-window character-level chunking with overlap."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        chunk = text[i : i + CHUNK_SIZE].strip()
        if len(chunk) > 80:   # skip tiny trailing chunks
            chunks.append(chunk)
        if i + CHUNK_SIZE >= len(text):
            break
    return chunks or [text[:CHUNK_SIZE]]


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_document(text: str, doc_id: str, metadata: Optional[dict] = None) -> int:
    """
    Embed and upsert a legal reference document into Pinecone.

    Args:
        text:     Full document text (e.g. Indian Contract Act section)
        doc_id:   Unique stable identifier (e.g. "ica-section-73")
        metadata: Extra fields stored alongside each chunk
                  (e.g. {"source": "Indian Contract Act", "type": "act", "section": "73"})

    Returns:
        Number of chunks ingested
    """
    if metadata is None:
        metadata = {}

    chunks = _chunk_text(text)
    embedder = _get_embedder()
    index = _get_index()

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        vectors.append({
            "id": f"{doc_id}__chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk, "doc_id": doc_id, **metadata},
        })

    # Upsert in batches of 100 (Pinecone limit)
    for start in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[start : start + 100])

    return len(chunks)


def delete_document(doc_id: str) -> None:
    """Delete all chunks for a given doc_id from the index."""
    index = _get_index()
    index.delete(filter={"doc_id": {"$eq": doc_id}})


def retrieve_context(query: str, top_k: int = 4) -> str:
    """
    Find the most relevant legal context chunks for a document query.

    Args:
        query:  Text to search against (first 512 chars used for speed)
        top_k:  Maximum number of chunks to return

    Returns:
        Concatenated context string, or "" if Pinecone is unavailable / empty
    """
    try:
        embedder = _get_embedder()
        index = _get_index()

        embedding = embedder.encode(query[:512]).tolist()
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

        matches = results.get("matches", [])
        if not matches:
            return ""

        lines = []
        for match in matches:
            score = match.get("score", 0)
            if score < 0.25:       # skip near-irrelevant results
                continue
            meta  = match.get("metadata", {})
            source = meta.get("source", "Legal Reference")
            text   = meta.get("text", "")
            lines.append(f"[{source}]: {text}")

        return "\n\n".join(lines)

    except Exception:
        # RAG is enhancement-only — never break the main pipeline
        return ""


def list_documents() -> list:
    """
    Return a deduplicated list of ingested document IDs and their metadata.
    Uses a dummy query to inspect index contents.
    """
    try:
        embedder = _get_embedder()
        index = _get_index()
        dummy = embedder.encode("legal contract").tolist()
        results = index.query(vector=dummy, top_k=100, include_metadata=True)
        seen = {}
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", ""),
                    "type":   meta.get("type", ""),
                }
        return list(seen.values())
    except Exception:
        return []
