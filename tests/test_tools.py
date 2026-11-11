from types import SimpleNamespace

import pytest

from rag import tools

CHUNKS = [
    {"id": 0, "source": "a.pdf", "kind": "pdf", "loc": 1, "part": 0, "text": "Raise prices in stages. Start with new customers."},
    {"id": 1, "source": "a.pdf", "kind": "pdf", "loc": 2, "part": 0, "text": "Annual billing lowers churn compared with monthly billing."},
    {"id": 2, "source": "a.pdf", "kind": "pdf", "loc": 2, "part": 1, "text": "Offer a payment plan as a downsell."},
    {"id": 3, "source": "b.epub", "kind": "epub", "loc": 4, "part": 0, "text": "word " * 400},
]


class FakeRetriever:
    """Stands in for rag.search.Retriever: no models, fixed ranking."""

    def __init__(self):
        self.chunks = CHUNKS
        self.calls = []

    def search(self, query, mode, top):
        self.calls.append((query, mode, top))
        return [SimpleNamespace(chunk_id=1, score=0.91234567), SimpleNamespace(chunk_id=0, score=0.5)][:top]


def test_search_returns_ranked_results_with_source_and_page():
    r = FakeRetriever()
    out = tools.search_books(r, "  how to lower churn  ", mode="hybrid", top=2)
    assert [x["rank"] for x in out] == [1, 2]
    assert out[0]["source"] == "a.pdf" and out[0]["page"] == 2 and out[0]["kind"] == "pdf"
    assert out[0]["score"] == 0.9123
    assert r.calls == [("how to lower churn", "hybrid", 2)]      # query is stripped, arguments passed through


def test_search_clips_long_passages_on_a_word_boundary():
    class OneHit(FakeRetriever):
        def search(self, query, mode, top):
            return [SimpleNamespace(chunk_id=3, score=1.0)]
    out = tools.search_books(OneHit(), "q", max_chars=50)
    assert len(out[0]["text"]) <= 50 + len(" ...")
    assert out[0]["text"].endswith(" ...")


def test_search_zero_max_chars_means_no_clipping():
    class OneHit(FakeRetriever):
        def search(self, query, mode, top):
            return [SimpleNamespace(chunk_id=3, score=1.0)]
    out = tools.search_books(OneHit(), "q", max_chars=0)
    assert out[0]["text"] == CHUNKS[3]["text"]


@pytest.mark.parametrize("kwargs, message", [
    ({"query": ""}, "query"),
    ({"query": "   "}, "query"),
    ({"query": "x", "mode": "magic"}, "mode"),
    ({"query": "x", "top": 0}, "top"),
    ({"query": "x", "top": tools.MAX_TOP + 1}, "top"),
])
def test_search_rejects_bad_arguments_with_a_helpful_message(kwargs, message):
    with pytest.raises(ValueError, match=message):
        tools.search_books(FakeRetriever(), **kwargs)


def test_get_page_joins_chunks_in_reading_order():
    out = tools.get_page(CHUNKS, "a.pdf", 2)
    assert out["chunks"] == 2
    assert out["text"].index("Annual billing") < out["text"].index("payment plan")


def test_get_page_unknown_page_raises():
    with pytest.raises(ValueError, match="no text found"):
        tools.get_page(CHUNKS, "a.pdf", 99)


def test_list_books_is_sorted_and_renames_fields():
    stats = [
        {"source": "b.pdf", "pages": 10, "no_text_pages": 1, "chunks": 20},
        {"source": "A.pdf", "pages": 5, "no_text_pages": 0, "chunks": 9},
    ]
    out = tools.list_books(stats)
    assert [x["source"] for x in out] == ["A.pdf", "b.pdf"]
    assert out[1] == {"source": "b.pdf", "pages": 10, "pages_without_text": 1, "chunks": 20}
