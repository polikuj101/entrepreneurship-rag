"""Evaluation metric kept free of model code so it can be tested on its own."""
from __future__ import annotations


def is_hit(chunks: list[dict], hits, source: str, page: int, k: int) -> bool:
    """True if any of the top-k hits comes from the gold source and the gold page.

    `hits` is any sequence of objects with a `chunk_id` attribute, best first.
    """
    return any(chunks[h.chunk_id]["source"] == source and chunks[h.chunk_id]["loc"] == page for h in hits[:k])
