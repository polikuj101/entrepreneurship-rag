import json
from pathlib import Path

QUESTIONS = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def load():
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_question_has_the_required_fields():
    for q in load():
        assert set(q) >= {"id", "question", "source", "page"}, q
        assert isinstance(q["page"], int) and q["page"] >= 1, q
        assert q["question"].strip().endswith("?"), q
        assert q["source"].lower().endswith(".pdf"), q   # gold pages are PDF pages, not EPUB sections


def test_ids_are_unique_and_questions_are_not_repeated():
    qs = load()
    assert len({q["id"] for q in qs}) == len(qs)
    assert len({q["question"].lower() for q in qs}) == len(qs)


def test_no_gold_page_carries_more_than_two_questions():
    qs = load()
    pairs = [(q["source"], q["page"]) for q in qs]
    # a page may carry two questions, but not three: that would let one page dominate the score
    assert max(pairs.count(p) for p in set(pairs)) <= 2
