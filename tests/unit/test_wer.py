import pytest

from conversation_deconvolution.core.types import Segment, Utterance
from conversation_deconvolution.evaluation.clustering_metrics import (
    conversation_metrics,
    labels_from_conversations,
)
from conversation_deconvolution.evaluation.wer import iou, match_by_iou, wer_report


def test_wer_known_values():
    assert wer_report([("the cat sat", "the cat sat")])["wer"] == 0.0
    r = wer_report([("the cat", "the dog")])
    assert r["wer"] == pytest.approx(0.5)
    r2 = wer_report([("a b c", "a x y")])
    assert r2["wer"] == pytest.approx(2 / 3)


def test_iou_partial():
    a = Utterance("a", None, 0, 2)
    b = Utterance("b", None, 1, 2)
    assert iou(a, b) == pytest.approx(0.5)


def test_match_by_iou_pairs_greedy_optimal():
    gt = [Utterance("g1", "A", 0, 2), Utterance("g2", "B", 3, 5)]
    pred = [Utterance("p1", "A", 0.1, 2), Utterance("p2", "B", 3.2, 5)]
    pairs = match_by_iou(gt, pred)
    assert [(p[0].id, p[1].id) for p in pairs] == [("g1", "p1"), ("g2", "p2")]


def test_identical_partitions_perfect():
    convs = [
        _conv("c1", [Utterance("u1", "A", 0, 1), Utterance("u2", "B", 1, 2)]),
        _conv("c2", [Utterance("u3", "C", 10, 11)]),
    ]
    keys = ["u1", "u2", "u3"]
    m = conversation_metrics(convs, convs, {k: k for k in keys})
    assert m["pairwise_f1"] == pytest.approx(1.0)
    assert m["ari"] == pytest.approx(1.0)
    assert m["nmi"] == pytest.approx(1.0)


def test_crossed_partitions_measurable():
    true = [
        _conv("c1", [_u("u1"), _u("u2"), _u("u3")]),
        _conv("c2", [_u("u4"), _u("u5")]),
    ]
    pred = [
        _conv("k1", [_u("u1"), _u("u2")]),
        _conv("k2", [_u("u3"), _u("u4"), _u("u5")]),
    ]
    keys = {f"u{i}": f"u{i}" for i in range(1, 6)}
    m = conversation_metrics(true, pred, keys)
    assert 0.0 < m["pairwise_f1"] < 1.0
    assert m["ari"] < 1.0


def _conv(cid, utts):
    from conversation_deconvolution.core.types import Conversation

    return Conversation(cid, sorted({u.speaker for u in utts}), utts)


def _u(uid):
    return Utterance(uid, "S", float(int(uid[1:])), float(int(uid[1:])) + 1)


def test_labels_from_conversations():
    convs = [_conv("c1", [_u("u1"), _u("u2")]), _conv("c2", [_u("u3")])]
    labels = labels_from_conversations(convs, ["u1", "u2", "u3"])
    assert labels.tolist() == [0, 0, 1]


def test_segment_iou():
    assert iou(Segment(0, 2), Segment(1, 3)) == pytest.approx(1 / 3)
