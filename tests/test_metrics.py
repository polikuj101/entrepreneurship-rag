from types import SimpleNamespace

from rag.metrics import is_hit

CHUNKS = [
    {"source": "a.pdf", "loc": 1},
    {"source": "a.pdf", "loc": 2},
    {"source": "b.pdf", "loc": 1},
    {"source": "b.pdf", "loc": 7},
]


def hits(*ids):
    return [SimpleNamespace(chunk_id=i) for i in ids]


def test_hit_when_gold_chunk_is_in_top_k():
    assert is_hit(CHUNKS, hits(2, 3, 0), "b.pdf", 7, k=2)


def test_miss_when_gold_chunk_is_below_k():
    assert not is_hit(CHUNKS, hits(0, 1, 3), "b.pdf", 7, k=2)
    assert is_hit(CHUNKS, hits(0, 1, 3), "b.pdf", 7, k=3)


def test_same_page_number_in_a_different_book_is_a_miss():
    assert not is_hit(CHUNKS, hits(0), "b.pdf", 1, k=1)      # a.pdf page 1 is not b.pdf page 1


def test_same_book_wrong_page_is_a_miss():
    assert not is_hit(CHUNKS, hits(1), "a.pdf", 1, k=1)


def test_no_hits_is_a_miss():
    assert not is_hit(CHUNKS, [], "a.pdf", 1, k=5)
