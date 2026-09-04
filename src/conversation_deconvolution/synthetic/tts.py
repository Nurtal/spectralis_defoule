import hashlib
import io
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = "rhasspy/piper-voices"
QUALITY = {"gilles": "low", "mls_1840": "low"}
DEFAULT_QUALITY = "medium"

EDGE_VOICE_MAP = {
    "siwis": "fr-FR-DeniseNeural",
    "tom": "fr-FR-RemyMultilingualNeural",
    "upmc": "fr-FR-EloiseNeural",
    "mls": "fr-FR-VivienneMultilingualNeural",
    "mls_1840": "fr-FR-HenriNeural",
    "gilles": "fr-BE-GerardNeural",
}


def create_tts(backend: str = "piper", **kwargs):
    if backend == "edge":
        return EdgeTts(**kwargs)
    return PiperTts(**kwargs)


class PiperTts:
    def __init__(self, cache_dir: str | Path = "models/piper", use_cuda: bool = False):
        self.cache_dir = Path(cache_dir)
        self.use_cuda = use_cuda
        self._voices: dict[str, object] = {}
        self.sample_cache = self.cache_dir / "samples"
        self.sample_cache.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        key = hashlib.md5(f"{voice_name}|{text}".encode()).hexdigest()
        cache_path = self.sample_cache / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path), 16000
        audio, sr = self._synthesize_raw(text, voice_name)
        np.save(cache_path, _resample_16k(audio, sr))
        return np.load(cache_path), 16000

    def _synthesize_raw(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        voice = self._ensure_voice(voice_name)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
            rate = wav_file.getframerate()
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())
            rate = wav_file.getframerate()
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate

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


class EdgeTts:
    def __init__(self, cache_dir: str | Path = "models/edge_tts"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_cache = self.cache_dir / "samples"
        self.sample_cache.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        key = hashlib.md5(f"{voice_name}|{text}".encode()).hexdigest()
        cache_path = self.sample_cache / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path), 16000
        audio = self._synthesize_raw(text, voice_name)
        np.save(cache_path, audio)
        return np.load(cache_path), 16000

    def _synthesize_raw(self, text: str, voice_name: str) -> np.ndarray:
        import edge_tts

        edge_voice = EDGE_VOICE_MAP.get(voice_name, voice_name)
        elapsed = time.monotonic() - self._last_call
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
            mp3_path = mp3_f.name
        try:
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(text, voice=edge_voice)
                    comm.save_sync(mp3_path)
                    self._last_call = time.monotonic()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            audio, sr = _load_mp3_via_ffmpeg(mp3_path)
        finally:
            Path(mp3_path).unlink(missing_ok=True)
        return _resample_16k(audio, sr)


def _load_mp3_via_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg", "-i", path, "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "pipe:1", "-loglevel", "error",
    ]
    raw = subprocess.check_output(cmd)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, sr


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    gcd = int(np.gcd(sr, 16000))
    out = resample_poly(audio, 16000 // gcd, sr // gcd)
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)
