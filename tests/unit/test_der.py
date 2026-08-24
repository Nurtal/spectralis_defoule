import pytest

from conversation_deconvolution.core.types import SpeakerTurn
from conversation_deconvolution.evaluation.der import DerResult, diarization_error_rate


def test_identical_turns_zero_der():
    ref = [SpeakerTurn("A", 0, 5), SpeakerTurn("B", 5, 10)]
    assert diarization_error_rate(ref, ref).der == pytest.approx(0.0, abs=1e-6)


def test_swapped_labels_zero_der():
    ref = [SpeakerTurn("A", 0, 5), SpeakerTurn("B", 5, 10)]
    hyp = [SpeakerTurn("X", 0, 5), SpeakerTurn("Y", 5, 10)]
    r = diarization_error_rate(ref, hyp)
    assert r.der == pytest.approx(0.0, abs=1e-6)
    assert r.correct == pytest.approx(9.0, abs=0.01)


def test_second_ref_mapped_away_is_confusion():
    ref = [SpeakerTurn("A", 0, 5), SpeakerTurn("B", 5, 10)]
    hyp = [SpeakerTurn("C", 0, 10)]
    r = diarization_error_rate(ref, hyp)
    assert r.correct == pytest.approx(4.5, abs=0.05)
    assert r.confusion == pytest.approx(4.5, abs=0.05)
    assert r.der == pytest.approx(0.5, abs=0.01)


def test_false_alarm_before_reference_speech():
    ref = [SpeakerTurn("A", 5, 10)]
    hyp = [SpeakerTurn("B", 0, 10)]
    r = diarization_error_rate(ref, hyp)
    assert r.false_alarm == pytest.approx(4.75, abs=0.05)
    assert r.confusion == 0.0
    assert r.der == pytest.approx(4.75 / 4.5, abs=0.01)


def test_collar_shrinks_reference():
    ref = [SpeakerTurn("A", 0, 1)]
    hyp = [SpeakerTurn("A", 0.25, 1.25)]
    r = diarization_error_rate(ref, hyp, collar=0.25)
    assert r.der == pytest.approx(0.0, abs=1e-6)
    assert r.total == pytest.approx(0.5, abs=1e-3)


def test_empty_hypothesis_all_miss_after_collar():
    ref = [SpeakerTurn("A", 0, 4), SpeakerTurn("B", 4, 6)]
    r = diarization_error_rate(ref, [])
    assert r.miss == pytest.approx(5.0, abs=0.05)
    assert r.total == pytest.approx(5.0, abs=0.05)
    assert r.der == pytest.approx(1.0, abs=1e-6)


def test_unmapped_extra_speaker_is_confusion():
    ref = [SpeakerTurn("A", 0, 4)]
    hyp = [SpeakerTurn("A", 0, 3), SpeakerTurn("B", 2.5, 3)]
    r = diarization_error_rate(ref, hyp)
    assert r.correct == pytest.approx(2.25, abs=0.06)
    assert r.confusion == pytest.approx(0.5, abs=0.06)
    assert r.false_alarm == 0.0
