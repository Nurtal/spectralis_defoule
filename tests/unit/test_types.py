import numpy as np
import pytest

from conversation_deconvolution.core.types import (
    Conversation,
    Segment,
    TranscriptResult,
    Utterance,
    conversation_from_dict,
    conversation_to_dict,
    result_from_dict,
    result_to_dict,
    utterance_from_dict,
    utterance_to_dict,
)


def test_segment_duration():
    assert Segment(1.0, 2.5).duration == 1.5


def test_utterance_round_trip():
    u = Utterance("u1", "speaker_01", 1.0, 2.5, "salut", 0.9, "fr")
    assert utterance_from_dict(utterance_to_dict(u), id_="u1") == u


def test_utterance_optional_fields_dropped():
    u = Utterance("u1", "speaker_01", 0.0, 1.0)
    d = utterance_to_dict(u)
    assert "confidence" not in d and "language" not in d


def test_conversation_schema_matches_readme():
    c = Conversation(
        "conversation_01",
        ["speaker_01"],
        [Utterance("u1", "speaker_01", 12.4, 15.2, "Tu viens demain ?")],
    )
    d = conversation_to_dict(c)
    assert set(d) == {"id", "participants", "utterances"}
    assert set(d["utterances"][0]) >= {"speaker", "start", "end", "text"}
    assert conversation_from_dict(d).id == "conversation_01"


def test_result_round_trip_preserves_sorting():
    c = Conversation(
        "c1",
        ["A", "B"],
        [
            Utterance("u2", "B", 3.0, 4.0, "b"),
            Utterance("u1", "A", 1.0, 2.0, "a"),
        ],
    )
    r = TranscriptResult(conversations=[c])
    r2 = result_from_dict(result_to_dict(r))
    assert [u.start for u in r2.utterances] == [1.0, 3.0]
    assert r2.conversations[0].participants == ["A", "B"]
