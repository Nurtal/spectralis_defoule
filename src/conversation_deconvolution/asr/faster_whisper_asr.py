import math
from dataclasses import dataclass

import numpy as np

from conversation_deconvolution.core.config import AsrConfig


@dataclass(frozen=True)
class AsrResult:
    text: str
    confidence: float
    language: str | None


class FasterWhisperAsr:
    def __init__(self, config: AsrConfig):
        self.cfg = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.cfg.model_size,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        return self._model

    def transcribe(self, segment: np.ndarray, language: str | None = None) -> AsrResult:
        model = self._ensure_model()
        audio = np.asarray(segment, dtype=np.float32)
        seg_iter, info = model.transcribe(
            audio,
            language=language or self.cfg.language,
            beam_size=1,
            vad_filter=False,
        )
        texts = []
        logprobs = []
        for seg in seg_iter:
            if seg.text.strip():
                texts.append(seg.text.strip())
                logprobs.append(seg.avg_logprob)
        confidence = float(np.mean([math.exp(p) for p in logprobs])) if logprobs else 0.0
        return AsrResult(" ".join(texts), confidence, info.language)
