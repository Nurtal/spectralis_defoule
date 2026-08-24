import numpy as np
from scipy.signal import butter, sosfilt


def place(clips: list[tuple[float, np.ndarray]], length: int) -> np.ndarray:
    mix = np.zeros(length, dtype=np.float64)
    for start_sec, audio in clips:
        i0 = round(start_sec * 16000)
        i1 = min(length, i0 + len(audio))
        if i1 > i0:
            mix[i0:i1] += audio[: i1 - i0]
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 1.0:
        mix /= peak
    return mix.astype(np.float32)


def _bandlimited_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(n)
    sos = butter(6, 3400.0, btype="low", fs=16000, output="sos")
    return sosfilt(sos, noise)


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    signal_power = float(np.mean(signal**2)) or 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = _bandlimited_noise(len(signal), rng)
    noise *= np.sqrt(noise_power / (float(np.mean(noise**2)) or 1e-12))
    return (signal + noise).astype(np.float32)
