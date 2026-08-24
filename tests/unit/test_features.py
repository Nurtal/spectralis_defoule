from conversation_deconvolution.conversation.features import (
    UnionFind,
    alternation,
    candidate_pairs,
    gap,
    temporal_score,
)
from conversation_deconvolution.core.types import Utterance


def U(i, s, t):
    return Utterance(i, s, t, t + 1.0)


def test_gap_clamps_negative():
    a = U("a", "A", 0.0)
    b = U("b", "B", 0.5)
    assert gap(a, b) == 0.0


def test_gap_positive():
    a = U("a", "A", 0.0)
    b = U("b", "B", 5.0)
    assert gap(a, b) == pytest_approx(4.0)


def pytest_approx(x):
    from pytest import approx

    return approx(x)


def test_alternation():
    a = U("a", "A", 0.0)
    b = U("b", "B", 1.0)
    c = U("c", None, 2.0)
    assert alternation(a, b) == 1.0
    assert alternation(a, a) == 0.0
    assert alternation(a, c) == 0.0


def test_temporal_score_decays():
    a = U("a", "A", 0.0)
    near = U("b", "B", 1.0)
    far = U("c", "B", 20.0)
    assert temporal_score(a, near, tau=4.0) > temporal_score(a, far, tau=4.0)


def test_candidate_pairs_strict_window():
    us = [U("u0", "A", 0), U("u1", "B", 1.5), U("u2", "A", 40)]
    assert candidate_pairs(us, max_gap=30.0) == [(0, 1)]


def test_union_find_groups():
    uf = UnionFind()
    uf.union(0, 1)
    uf.union(1, 2)
    groups = uf.groups()
    assert sorted(groups[uf.find(0)]) == [0, 1, 2]
    assert uf.find(3) != uf.find(0)
