import io
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = "rhasspy/piper-voices"
QUALITY = {"gilles": "low"}
DEFAULT_QUALITY = "medium"


class PiperTts:
    def __init__(self, cache_dir: str | Path = "models/piper", use_cuda: bool = False):
        self.cache_dir = Path(cache_dir)
        self.use_cuda = use_cuda
        self._voices: dict[str, object] = {}

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        voice = self._ensure_voice(voice_name)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        rate = wav_file.getframerate()
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())
            rate = wav_file.getframerate()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return _resample_16k(audio, rate), 16000

    def _ensure_voice(self, name: str):
        if name not in self._voices:
            quality = QUALITY.get(name, DEFAULT_QUALITY)
            base = f"fr/fr_FR/{name}/{quality}/fr_FR-{name}-{quality}"
            model_path = hf_hub_download(REPO, base + ".onnx", cache_dir=self.cache_dir)
            config_path = hf_hub_download(REPO, base + ".onnx.json", cache_dir=self.cache_dir)
            from piper import PiperVoice

            self._voices[name] = PiperVoice.load(
                model_path, config_path=config_path, use_cuda=self.use_cuda
            )
        return self._voices[name]


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    gcd = int(np.gcd(sr, 16000))
    out = resample_poly(audio, 16000 // gcd, sr // gcd)
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)
