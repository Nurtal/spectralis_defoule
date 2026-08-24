import numpy as np

from conversation_deconvolution.core.config import VadConfig
from conversation_deconvolution.core.types import Segment, VadResult


class SileroVad:
    def __init__(self, config: VadConfig):
        self.cfg = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    @property
    def frame_rate(self) -> float:
        return 16000.0 / 512.0

    def detect(self, audio: np.ndarray) -> VadResult:
        from silero_vad import get_speech_timestamps

        model = self._ensure_model()
        audio = np.asarray(audio, dtype=np.float32)
        stamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=16000,
            speech_pad_ms=0,
            min_speech_duration_ms=self.cfg.min_speech_ms,
            min_silence_duration_ms=self.cfg.min_silence_ms,
            threshold=self.cfg.threshold,
        )
        segments = [Segment(s["start"] / 16000.0, s["end"] / 16000.0) for s in stamps]
        probs = self.frame_probs(audio)
        return VadResult(segments=segments, frame_probs=probs, frame_rate=self.frame_rate)

    def frame_probs(self, audio: np.ndarray) -> np.ndarray:
        import torch

        model = self._ensure_model()
        window = 512
        n = len(audio) // window
        probs = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for k in range(n):
                chunk = torch.from_numpy(audio[k * window : (k + 1) * window])
                probs[k] = float(model(chunk, 16000).item())
        return probs
