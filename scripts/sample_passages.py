"""Print a reproducible random sample of PDF pages so evaluation questions can be written from them.

Writes nothing under version control: the output goes to stdout only (book text is copyrighted).
"""
import argparse
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=45)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--min-words", type=int, default=120)
ap.add_argument("--max-words", type=int, default=180)
args = ap.parse_args()

pages = defaultdict(list)
for line in open("data/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    if c["kind"] == "pdf":
        pages[(c["source"], c["loc"])].append(c)

cands = []
for key, parts in pages.items():
    text = " ".join(p["text"] for p in sorted(parts, key=lambda p: p["part"]))
    w = len(text.split())
    if args.min_words <= w:
        cands.append((key, " ".join(text.split()[: args.max_words])))

random.Random(args.seed).shuffle(cands)
# at most 4 pages per book, so one long book does not dominate
per_book, picked = defaultdict(int), []
for (src, page), text in cands:
    if per_book[src] >= 4:
        continue
    per_book[src] += 1
    picked.append((src, page, text))
    if len(picked) == args.n:
        break

for i, (src, page, text) in enumerate(picked, 1):
    print(f"### {i} | {src} | p.{page}\n{text}\n")
