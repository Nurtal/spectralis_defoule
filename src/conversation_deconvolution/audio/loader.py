from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr == target_sr:
        return mono.astype(np.float32)
    gcd = np.gcd(sr, target_sr)
    up, down = target_sr // gcd, sr // gcd
    resampled = resample_poly(mono, up, down).astype(np.float32)
    peak = float(np.max(np.abs(resampled))) if len(resampled) else 0.0
    if peak > 1.0:
        resampled /= peak
    return resampled
