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

    def transcribe(
        self,
        segment: np.ndarray,
        language: str | None = None,
        speaker_context: str | None = None,
    ) -> AsrResult:
        model = self._ensure_model()
        audio = np.asarray(segment, dtype=np.float32)
        beam_size = self.cfg.beam_size
        initial_prompt = self.cfg.initial_prompt
        if speaker_context and self.cfg.use_speaker_beam:
            prompt = f"{speaker_context} " if not initial_prompt else f"{initial_prompt} {speaker_context}"
            initial_prompt = prompt
        seg_iter, info = model.transcribe(
            audio,
            language=language or self.cfg.language,
            beam_size=beam_size,
            vad_filter=False,
            initial_prompt=initial_prompt,
        )
        texts = []
        logprobs = []
        for seg in seg_iter:
            if seg.text.strip():
                texts.append(seg.text.strip())
                logprobs.append(seg.avg_logprob)
        confidence = float(np.mean([math.exp(p) for p in logprobs])) if logprobs else 0.0
        return AsrResult(" ".join(texts), confidence, info.language)
