import numpy as np
import soundfile as sf

from conversation_deconvolution.audio.loader import load_audio


def test_load_resamples_to_mono_16k(tmp_path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    data = np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    p = tmp_path / "a.wav"
    sf.write(p, data, sr)
    y = load_audio(p)
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - 16000) < 160


def test_load_native_sr_passthrough(tmp_path):
    t = np.arange(8000) / 16000
    p = tmp_path / "b.wav"
    sf.write(p, np.sin(2 * np.pi * 100 * t).astype(np.float32), 16000)
    assert len(load_audio(p)) == 8000
