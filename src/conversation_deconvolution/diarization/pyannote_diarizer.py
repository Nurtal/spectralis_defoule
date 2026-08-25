import numpy as np
import torch

from conversation_deconvolution.core.types import SpeakerTurn


class PyannoteDiarizer:
    def __init__(
        self, token: str, model: str = "pyannote/speaker-diarization-3.1", device: str = "cuda"
    ):
        from pyannote.audio import Pipeline as _Pipeline

        self._pipe = _Pipeline.from_pretrained(model, token=token)
        self._pipe = self._pipe.to(torch.device(device))
        self.overlap_regions_: list[tuple[float, float]] = []

    def diarize(self, audio: np.ndarray) -> tuple[list[SpeakerTurn], list[np.ndarray]]:
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        result = self._pipe(
            {"waveform": waveform, "sample_rate": 16000},
            num_speakers=None,
        )
        ann = result.speaker_diarization
        turns = []
        for seg, _, speaker in ann.itertracks(yield_label=True):
            if seg.end - seg.start >= 0.3:
                turns.append(SpeakerTurn(speaker, seg.start, seg.end))
        self.overlap_regions_ = []
        return turns, []
