"""Tool logic for the MCP server, kept free of MCP and model imports so it can be unit-tested on its own.

Every function takes the data it needs as arguments (a retriever-like object, the chunk list, the stats list).
Errors are raised as ValueError with a message an agent can act on.
"""
from __future__ import annotations

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
MAX_TOP = 20


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."


def search_books(retriever, query: str, mode: str = "hybrid_rerank", top: int = 5, max_chars: int = 600) -> list[dict]:
    """Search the indexed books. `retriever` needs `.chunks` and `.search(query, mode=, top=)` returning hits with
    `chunk_id` and `score`."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    if not 1 <= top <= MAX_TOP:
        raise ValueError(f"top must be between 1 and {MAX_TOP}; got {top}")
    hits = retriever.search(query.strip(), mode=mode, top=top)
    out = []
    for rank, h in enumerate(hits, start=1):
        c = retriever.chunks[h.chunk_id]
        out.append({
            "rank": rank,
            "source": c["source"],
            "page": c["loc"],              # PDF page number, or section number for EPUB files
            "kind": c["kind"],
            "score": round(float(h.score), 4),
            "text": _clip(c["text"], max_chars),
        })
    return out


def get_page(chunks: list[dict], source: str, page: int, max_chars: int = 4000) -> dict:
    """Return all text of one page (PDF) or section (EPUB), joined in reading order."""
    parts = sorted((c for c in chunks if c["source"] == source and c["loc"] == page), key=lambda c: c["part"])
    if not parts:
        known = sorted({c["source"] for c in chunks})
        raise ValueError(f"no text found for source={source!r} page={page}. Known sources start with: {known[:3]}")
    # chunks overlap by design, so joining them repeats a few words at each seam; that is acceptable for reading
    text = "\n".join(c["text"] for c in parts)
    return {"source": source, "page": page, "kind": parts[0]["kind"], "chunks": len(parts), "text": _clip(text, max_chars)}


def list_books(stats: list[dict]) -> list[dict]:
    """One row per indexed file with page and chunk counts (from data/ingest_stats.json)."""
    return [
        {"source": s["source"], "pages": s["pages"], "pages_without_text": s["no_text_pages"], "chunks": s["chunks"]}
        for s in sorted(stats, key=lambda s: s["source"].lower())
    ]
