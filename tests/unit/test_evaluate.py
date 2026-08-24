
from conversation_deconvolution.cli import evaluate_results
from conversation_deconvolution.core.types import Conversation, TranscriptResult


def _conv(cid, utts):
    return Conversation(id=cid, participants=[], utterances=utts)


def test_dense_interleaved_grouping_still_evaluated():
    # Every GT utterance overlaps another one across threads: the old
    # non-overlap-only matching collapsed to zero pairs and zeroed F1.
    def U(uid, s, e, text):
        from conversation_deconvolution.core.types import Utterance

        return Utterance(id=uid, speaker="S", start=s, end=e, text=text)

    ref = TranscriptResult(
        utterances=[],
        overlaps=[],
        conversations=[
            _conv(
                "c1",
                [U("a1", 0.0, 2.0, "bonjour ici"), U("a2", 2.5, 4.5, "a plus tard")],
            ),
            _conv(
                "c2",
                [U("b1", 0.5, 2.5, "salut la bas"), U("b2", 3.0, 5.0, "bonne soiree")],
            ),
        ],
    )
    hyp = TranscriptResult(
        utterances=[],
        overlaps=[],
        conversations=[
            _conv("k1", [U("p1", 0.0, 2.0, "bonjour ici"), U("p2", 2.5, 4.5, "a plus tard")]),
            _conv("k2", [U("p3", 0.5, 2.5, "salut la bas"), U("p4", 3.0, 5.0, "bonne soiree")]),
        ],
    )
    m = evaluate_results(ref, hyp)
    assert m["pairwise_F1"] == 1.0
    assert m["ARI"] == 1.0


def test_conversation_metrics_id_mapping():
    from conversation_deconvolution.evaluation.clustering_metrics import (
        conversation_metrics,
    )

    def U(uid, s, e):
        from conversation_deconvolution.core.types import Utterance

        return Utterance(id=uid, speaker="S", start=s, end=e, text="x")

    ref = [_conv("c1", [U("a1", 0, 1), U("a2", 2, 3)]), _conv("c2", [U("b1", 4, 5)])]
    hyp = [_conv("k1", [U("p1", 0, 1)]), _conv("k2", [U("p2", 2, 3), U("p3", 4, 5)])]
    m = conversation_metrics(ref, hyp, {"a1": "p1", "a2": "p2", "b1": "p3"})
    # only true pair (a1,a2) is split across predicted conversations: tp=0
    assert m["pairwise_f1"] == 0.0
    # both GT utterances of c1 matched to predicted utterances of k1: perfect
    m2 = conversation_metrics(ref, hyp, {"a1": "p1", "a2": "p1", "b1": "p3"})
    assert m2["pairwise_f1"] == 1.0
