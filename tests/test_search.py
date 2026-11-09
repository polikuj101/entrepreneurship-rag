import pytest

from rag.search import rrf_fuse, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Raise Prices, don't lose 50% of clients!") == ["raise", "prices", "don", "t", "lose", "50", "of", "clients"]


def test_rrf_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_single_ranking_keeps_order():
    assert [c for c, _ in rrf_fuse([[7, 3, 9]])] == [7, 3, 9]


def test_rrf_chunk_in_both_rankings_beats_chunks_in_one():
    fused = rrf_fuse([[1, 2, 3], [4, 2, 5]])
    assert fused[0][0] == 2                      # appears in both lists
    assert {c for c, _ in fused} == {1, 2, 3, 4, 5}


def test_rrf_score_matches_formula():
    k = 60
    fused = dict(rrf_fuse([[10, 20], [20]], k=k))
    assert fused[10] == pytest.approx(1 / (k + 1))
    assert fused[20] == pytest.approx(1 / (k + 2) + 1 / (k + 1))


def test_rrf_ties_are_deterministic_in_first_seen_order():
    # 1 and 2 each win exactly one ranking at the same rank, so their scores are equal
    first = [c for c, _ in rrf_fuse([[1], [2]])]
    second = [c for c, _ in rrf_fuse([[1], [2]])]
    assert first == second == [1, 2]
