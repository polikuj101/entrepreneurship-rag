"""Dense, BM25, hybrid (reciprocal rank fusion) and reranked retrieval over the chunk file."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(path: str = "data/chunks.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass
class Hit:
    chunk_id: int
    score: float


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Each ranking is a list of chunk ids, best first.

    A chunk's score is the sum of 1 / (k + rank) over the rankings it appears in (rank starts at 1).
    Ties keep the order in which chunks were first seen, so the result is deterministic.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class Retriever:
    def __init__(self, chunks_path: str = "data/chunks.jsonl", emb_path: str = "data/embeddings.npy"):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = load_chunks(chunks_path)
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        emb_file = Path(emb_path)
        if emb_file.exists():
            self.emb = np.load(emb_file)
        else:
            self.emb = self.embedder.encode(
                [c["text"] for c in self.chunks], batch_size=64, normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            np.save(emb_file, self.emb)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._reranker = None

    # --- single retrievers -------------------------------------------------
    def dense(self, query: str, k: int = 50) -> list[Hit]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Hit(int(i), float(scores[i])) for i in idx]

    def sparse(self, query: str, k: int = 50) -> list[Hit]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Hit(int(i), float(scores[i])) for i in idx]

    # --- fusion and reranking ---------------------------------------------
    def hybrid(self, query: str, k: int = 50) -> list[Hit]:
        rankings = [[h.chunk_id for h in self.dense(query, k)], [h.chunk_id for h in self.sparse(query, k)]]
        return [Hit(i, s) for i, s in rrf_fuse(rankings)[:k]]

    def rerank(self, query: str, candidates: list[Hit], top: int = 10) -> list[Hit]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=512)
        pairs = [(query, self.chunks[h.chunk_id]["text"]) for h in candidates]
        scores = self._reranker.predict(pairs, batch_size=16)
        order = np.argsort(-scores)[:top]
        return [Hit(candidates[i].chunk_id, float(scores[i])) for i in order]

    def search(self, query: str, mode: str = "hybrid_rerank", top: int = 10, pool: int = 30) -> list[Hit]:
        if mode == "dense":
            return self.dense(query, top)
        if mode == "bm25":
            return self.sparse(query, top)
        if mode == "hybrid":
            return self.hybrid(query, top)
        if mode == "hybrid_rerank":
            return self.rerank(query, self.hybrid(query, pool), top)
        raise ValueError(mode)


if __name__ == "__main__":
    import sys

    r = Retriever()
    q = " ".join(sys.argv[1:]) or "how do I price my offer"
    for h in r.search(q, top=5):
        c = r.chunks[h.chunk_id]
        print(f"{h.score:.3f}  {c['source']}  p.{c['loc']}\n    {c['text'][:160]}...")
