"""Parse PDFs and EPUBs into page-level chunks.

Chunks never cross a page (PDF) or section (EPUB) boundary, so every chunk has an exact
source location. That is what the evaluation uses as ground truth.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

CHUNK_WORDS = 200
OVERLAP_WORDS = 40
MIN_PAGE_CHARS = 50  # pages with less text than this are counted as "no text layer"


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)      # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pdf(path: Path):
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        yield i, clean(page.get_text("text"))


def parse_epub(path: Path):
    from bs4 import BeautifulSoup
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    n = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        n += 1
        soup = BeautifulSoup(item.get_content(), "html.parser")
        yield n, clean(soup.get_text(" "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("books_dir")
    ap.add_argument("--out", default="data/chunks.jsonl")
    args = ap.parse_args()

    root = Path(args.books_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".pdf", ".epub"})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = []
    n_chunks = 0
    with out_path.open("w", encoding="utf-8") as out:
        for f in files:
            rel = f.relative_to(root).as_posix()
            pages = parse_pdf(f) if f.suffix.lower() == ".pdf" else parse_epub(f)
            total = empty = chunks = 0
            for loc, text in pages:
                total += 1
                if len(text) < MIN_PAGE_CHARS:
                    empty += 1
                    continue
                for j, piece in enumerate(split_words(text)):
                    out.write(json.dumps({
                        "id": n_chunks, "source": rel, "kind": f.suffix.lower()[1:],
                        "loc": loc, "part": j, "text": piece,
                    }, ensure_ascii=False) + "\n")
                    n_chunks += 1
                    chunks += 1
            stats.append({"source": rel, "pages": total, "no_text_pages": empty, "chunks": chunks})
            print(f"{rel}: {total} pages, {empty} without text, {chunks} chunks")

    Path(out_path.parent / "ingest_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"total chunks: {n_chunks}")


if __name__ == "__main__":
    main()
