import numpy as np
import pytest

from conversation_deconvolution.synthetic.mixer import add_noise, place


def sine(freq, dur, amp=0.5):
    t = np.arange(int(dur * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_place_adds_without_overlap_loss():
    total = 32000
    a = sine(220, 0.5)
    b = sine(440, 0.5)
    mix = place([(0.0, a), (1.0, b)], total)
    assert len(mix) == total
    assert abs(mix[4000] - a[4000]) < 1e-6
    assert abs(mix[20000] - b[4000]) < 1e-6


def test_place_overlapping_sums():
    total = 16000
    a = sine(220, 1.0)
    mix = place([(0.0, a), (0.0, a)], total)
    assert abs(mix[100] - 2 * a[100]) < 1e-6


def test_add_noise_exact_snr():
    sig = sine(220, 1.0)
    noisy = add_noise(sig, snr_db=10.0, rng=np.random.default_rng(0))
    noise = noisy - sig
    snr = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=0.5)


def test_add_noise_zero_db():
    sig = sine(330, 1.0)
    noisy = add_noise(sig, snr_db=0.0, rng=np.random.default_rng(3))
    noise = noisy - sig
    snr = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
    assert snr == pytest.approx(0.0, abs=0.5)
