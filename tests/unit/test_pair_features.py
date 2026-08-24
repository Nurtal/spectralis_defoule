import math

import pytest

from conversation_deconvolution.conversation.pair_features import (
    pair_feature_names,
    pair_features,
)
from conversation_deconvolution.core.types import Utterance


def U(uid, spk, start, end):
    return Utterance(uid, spk, start, end)


def test_disjoint_pair_features():
    a = U("u1", "A", 0.0, 1.0)
    b = U("u2", "B", 3.0, 4.0)
    v = pair_features(a, b, 0, 5, 0.25, tau=4.0)
    assert v[0] == pytest.approx(2.0)
    assert v[1] == pytest.approx(math.log(3.0))
    assert v[2] == pytest.approx(math.exp(-0.5))
    assert v[3] == 1.0
    assert v[4] == 0.0
    assert v[5] == 0.0
    assert v[6] == 0.25
    assert v[7] == 5.0
    assert v[8] == 1.0


def test_overlapping_pair_clamps_gap_and_ratios_overlap():
    a = U("u1", "A", 0.0, 2.0)
    b = U("u2", "B", 1.0, 3.0)
    v = pair_features(a, b, 0, 1, -0.1, tau=1.0)
    assert v[0] == 0.0
    assert v[1] == 0.0
    assert v[2] == 1.0
    assert v[5] == pytest.approx(0.5)


def test_same_speaker_and_none_speaker():
    a = U("u1", "A", 0.0, 1.0)
    b = U("u2", "A", 2.0, 3.0)
    c = U("u3", None, 2.0, 3.0)
    assert pair_features(a, b, 0, 1, 0.0, 1.0)[4] == 1.0
    assert pair_features(a, b, 0, 1, 0.0, 1.0)[3] == 0.0
    assert pair_features(a, c, 0, 1, 0.0, 1.0)[4] == 0.0


def test_names_match_vector_length():
    a = U("u", "A", 0.0, 1.0)
    b = U("v", "B", 1.0, 2.0)
    assert len(pair_feature_names()) == len(pair_features(a, b, 0, 1, 0.0, 1.0))
