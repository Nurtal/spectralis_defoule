import numpy as np

from conversation_deconvolution.core.types import SpeakerTurn
from conversation_deconvolution.diarization.timeline import merge_turns


class SpeakerDiarizer:
    def __init__(self, vad, embedder, clusterer, config):
        self.vad = vad
        self.embedder = embedder
        self.clusterer = clusterer
        self.cfg = config

    def diarize(self, audio: np.ndarray) -> tuple[list[SpeakerTurn], list[np.ndarray]]:
        result = self.vad.detect(audio)
        kept_segments = []
        embeddings = []
        for seg in result.segments:
            if seg.duration < self.cfg.min_segment_sec:
                continue
            s, e = int(seg.start * 16000), int(seg.end * 16000)
            kept_segments.append(seg)
            embeddings.append(self.embedder.embed(audio[s:e]))
        if not kept_segments:
            return [], []
        labels = self.clusterer.fit_predict(
            np.array(embeddings), n_speakers=self.cfg.num_speakers
        )
        turns = [
            SpeakerTurn(f"SPEAKER_{int(label):02d}", seg.start, seg.end)
            for seg, label in zip(kept_segments, labels)
        ]
        return merge_turns(turns), embeddings
