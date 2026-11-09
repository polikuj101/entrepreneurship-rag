# Entrepreneurship RAG: hybrid retrieval over a small book corpus, with a measured Hit Rate

A retrieval pipeline over 19 business books (17 PDFs, 2 EPUBs): page-level chunking, dense embeddings, BM25, reciprocal rank fusion, and a cross-encoder reranker. The repository includes a 34-question evaluation set and the script that measures Hit@k for each retrieval mode.

**Author:** Nurbolat (Bo) Mirash, [polikuj101](https://github.com/polikuj101)

## Status, stated plainly

- **This is a rebuild.** I built an earlier version of this project and lost the code. This repository is a new implementation of the same idea, written in November 2026. Numbers from the earlier version are not claimed here.
- **Retrieval only.** There is no answer-generation step in this repository. It finds passages; it does not write answers.
- **Small, optimistic evaluation.** 34 questions, written by an AI coding assistant (Claude Code) from randomly sampled pages. I have not independently reviewed every question and gold page. The questions share vocabulary with their source pages, which favors keyword retrieval and makes every score optimistic. See [Limitations](#limitations).
- **The books are not included.** They are copyrighted, and several of them are marked "not for distribution". You need your own copies to reproduce the numbers. The repository contains code, evaluation questions with page references, and results only.
- **Who wrote the code:** AI coding assistants wrote most of the implementation from my specifications. I chose the goal and the design; the numbers below come from running the scripts in this repository on my laptop.

## Results

Measured by `eval/run_eval.py` on 34 questions. A question is a hit at k if one of the top-k chunks comes from the gold book and the gold page.

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Latency per query (CPU) |
|---|---|---|---|---|---|
| Dense (bge-small) | 50% (17/34) | 79% (27/34) | 88% (30/34) | 94% (32/34) | 16 ms |
| BM25 | 68% (23/34) | 85% (29/34) | 94% (32/34) | 94% (32/34) | 25 ms |
| Hybrid (RRF of dense + BM25) | 65% (22/34) | 91% (31/34) | 97% (33/34) | 100% (34/34) | 48 ms |
| Hybrid + reranker (bge-reranker-base) | **91% (31/34)** | **100% (34/34)** | 100% (34/34) | 100% (34/34) | 6,994 ms |

What I take from this, and what I do not:

- Fusing dense and BM25 beat either alone at Hit@3 (91% against 79% and 85%). With 34 questions, one question is about 3 points, so differences of a few points are not meaningful.
- The reranker moved Hit@1 from 65% to 91%, which is the largest effect in the table. It is also about 150 times slower on CPU, because it scores 30 candidate passages per query with a cross-encoder. On a GPU this would be much faster; I did not measure it.
- 100% Hit@3 on 34 questions does not mean 100% on real use. See below.

## How it works

```
books/ (not in repo) -> rag/ingest.py -> data/chunks.jsonl
                                              |
                       bge-small embeddings   |   BM25 index
                                  \           |          /
                                   rank fusion (RRF, k=60)
                                              |
                          top 30 -> bge-reranker-base -> top 10
```

| Part | Choice | Why |
|---|---|---|
| Parsing | PyMuPDF for PDFs, ebooklib for EPUBs | No paid API; runs offline |
| Chunks | About 200 words, 40-word overlap, never crossing a page or EPUB section | Every chunk has an exact location, which the evaluation needs |
| Dense | `BAAI/bge-small-en-v1.5`, cosine similarity in NumPy | Small enough for a laptop CPU |
| Sparse | `rank-bm25` (BM25Okapi) over lowercase word tokens | Strong on exact terms and names |
| Fusion | Reciprocal rank fusion, k = 60 | No score calibration between dense and BM25 needed |
| Rerank | `BAAI/bge-reranker-base` cross-encoder on the top 30 | Re-scores query and passage together |

The corpus: 19 files, 1,911 pages, 5,317 chunks, about 919,000 words. Of these pages, 147 had no extractable text (scanned or image-only) and were skipped. One PDF, a 12-page SaaS playbook, has no text layer at all and contributes nothing.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python rag/ingest.py /path/to/your/books          # writes data/chunks.jsonl
python rag/search.py "how should I raise prices"  # builds embeddings on first run (several minutes on CPU)
python eval/run_eval.py                           # writes eval/results.md
```

The evaluation questions refer to specific file names and pages, so the numbers only reproduce if you have the same files. The books I used are listed in `eval/questions.jsonl` by path. Page numbers are PDF page indexes, not printed page numbers.

## Limitations

- **34 questions is a small sample.** The scores have wide uncertainty, and I did not compute confidence intervals.
- **The questions favor the retrievers.** They were written from the target page, so they reuse its words. Real users paraphrase more. Dense retrieval is likely to gain relative to BM25 on real queries; I did not test that.
- **One gold page per question.** Several books cover the same topics, such as raising prices, ad hooks and show rates. A good answer from a different page counts as a miss here, so the true quality may be higher than the table shows. Some questions may also be answerable from a second page I did not label.
- **Repeated footers add noise.** Several PDFs repeat a copyright footer on every page. I did not strip it.
- **Mostly one author's books.** 12 of 19 files are playbooks by the same author, which means near-duplicate topics and a style that is easy to retrieve.
- **CPU latency only.** The 7-second reranker figure was measured on a laptop CPU. Throughput under load was not tested.
- **No answer quality.** Nothing here measures whether a language model would answer correctly from the retrieved passages.

## Layout

```
rag/ingest.py          parse and chunk
rag/search.py          dense, BM25, hybrid, rerank
eval/questions.jsonl   34 questions with gold book and page
eval/run_eval.py       Hit@k for each mode
eval/results.md        output of the last run
scripts/sample_passages.py   seeded sampler used to pick pages for question writing
```
