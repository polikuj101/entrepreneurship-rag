"""Hit Rate@k for each retrieval mode against eval/questions.jsonl.

A question counts as a hit at k if any of the top-k chunks comes from the gold source and the
gold page (PDF) -- so ground truth is page-level, not chunk-level.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.search import Retriever  # noqa: E402

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
KS = [1, 3, 5, 10]


def main() -> None:
    qs = [json.loads(line) for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    r = Retriever()
    rows = []
    misses = {}
    for mode in MODES:
        hits = {k: 0 for k in KS}
        t0 = time.perf_counter()
        for q in qs:
            res = r.search(q["question"], mode=mode, top=max(KS))
            for k in KS:
                if any(r.chunks[h.chunk_id]["source"] == q["source"] and r.chunks[h.chunk_id]["loc"] == q["page"]
                       for h in res[:k]):
                    hits[k] += 1
            if not any(r.chunks[h.chunk_id]["source"] == q["source"] and r.chunks[h.chunk_id]["loc"] == q["page"]
                       for h in res[:3]):
                misses.setdefault(mode, []).append(q["id"])
        ms = (time.perf_counter() - t0) / len(qs) * 1000
        rows.append((mode, {k: hits[k] / len(qs) for k in KS}, ms))

    n = len(qs)
    lines = [f"Questions: {n}\n", "| mode | " + " | ".join(f"Hit@{k}" for k in KS) + " | ms/query (CPU) |",
             "|---|" + "---|" * (len(KS) + 1)]
    for mode, h, ms in rows:
        lines.append(f"| {mode} | " + " | ".join(f"{h[k]:.0%} ({round(h[k]*n)}/{n})" for k in KS) + f" | {ms:.0f} |")
    out = "\n".join(lines)
    print(out)
    Path("eval/results.md").write_text(out + "\n", encoding="utf-8")
    Path("eval/misses_at_3.json").write_text(json.dumps(misses, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
