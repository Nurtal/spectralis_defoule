import numpy as np

from conversation_deconvolution.core.types import Segment
from conversation_deconvolution.evaluation.vad_metrics import vad_prf

RATE = 50.0
N = 500


def test_perfect_alignment():
    probs = np.zeros(N)
    probs[100:300] = 0.9
    m = vad_prf(probs, RATE, [Segment(2.0, 6.0)])
    assert m["f1"] == pytest_approx(1.0)


def test_no_detection():
    probs = np.zeros(N)
    m = vad_prf(probs, RATE, [Segment(2.0, 6.0)])
    assert m["recall"] == 0.0 and m["precision"] == 0.0


def test_partial_coverage():
    probs = np.zeros(N)
    probs[150:350] = 0.9
    m = vad_prf(probs, RATE, [Segment(2.0, 6.0)])
    assert m["recall"] == pytest_approx(0.75)
    assert m["precision"] == pytest_approx(0.75)


def pytest_approx(x):
    from pytest import approx

    return approx(x, abs=1e-6)
