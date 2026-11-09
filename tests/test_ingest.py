import json
import sys

import fitz  # PyMuPDF

from rag import ingest


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_split_words_empty_text_gives_no_chunks():
    assert ingest.split_words("") == []
    assert ingest.split_words("   \n  ") == []


def test_split_words_short_text_is_one_chunk():
    assert ingest.split_words("one two three") == ["one two three"]


def test_split_words_chunks_respect_size_and_overlap():
    text = words(500)
    chunks = ingest.split_words(text, size=200, overlap=40)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 200 for c in chunks)
    # consecutive chunks share exactly `overlap` words
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-40:] == b.split()[:40]


def test_split_words_covers_every_word_once_in_order():
    text = words(333)
    chunks = ingest.split_words(text, size=100, overlap=10)
    rebuilt: list[str] = []
    for i, c in enumerate(chunks):
        ws = c.split()
        rebuilt.extend(ws if i == 0 else ws[10:])
    assert rebuilt == text.split()


def test_clean_rejoins_hyphenated_line_breaks_and_collapses_whitespace():
    assert ingest.clean("exam-\nple   of\n\ntext") == "example of text"
    assert ingest.clean("soft­hyphen") == "softhyphen"


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_parse_pdf_numbers_pages_from_one(tmp_path):
    pdf = tmp_path / "book.pdf"
    make_pdf(pdf, ["first page text", "second page text"])
    got = list(ingest.parse_pdf(pdf))
    assert [loc for loc, _ in got] == [1, 2]
    assert "first page" in got[0][1]


def test_main_skips_pages_without_text_and_keeps_locations(tmp_path, monkeypatch):
    books = tmp_path / "books"
    books.mkdir()
    long_text = "Pricing power matters because customers compare offers before they decide to buy."
    make_pdf(books / "a.pdf", [long_text, "", long_text])
    out = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(sys, "argv", ["ingest", str(books), "--out", str(out)])

    ingest.main()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["loc"] for r in rows] == [1, 3]          # page 2 has no text and is skipped
    assert all(r["source"] == "a.pdf" and r["kind"] == "pdf" for r in rows)
    assert [r["id"] for r in rows] == [0, 1]
    stats = json.loads((tmp_path / "ingest_stats.json").read_text(encoding="utf-8"))
    assert stats[0]["pages"] == 3 and stats[0]["no_text_pages"] == 1 and stats[0]["chunks"] == 2
