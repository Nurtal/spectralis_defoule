import numpy as np

from conversation_deconvolution.core.types import SpeakerTurn
from conversation_deconvolution.diarization.timeline import (
    make_windows,
    margin_regions,
    merge_segments,
    merge_turns,
    overlap_regions,
    windows_to_turns,
)


class SpeakerDiarizer:
    def __init__(self, vad, embedder, clusterer, config):
        self.vad = vad
        self.embedder = embedder
        self.clusterer = clusterer
        self.cfg = config

    def diarize(self, audio: np.ndarray) -> tuple[list[SpeakerTurn], list[np.ndarray]]:
        result = self.vad.detect(audio)
        segments = [s for s in result.segments if s.duration >= self.cfg.min_segment_sec]
        windows = make_windows(segments, self.cfg.window_sec, self.cfg.hop_sec)
        if not windows:
            return [], []
        embeddings = [
            self.embedder.embed(audio[int(s * 16000) : int(e * 16000)]) for s, e in windows
        ]
        labels = self.clusterer.fit_predict(
            np.array(embeddings), n_speakers=self.cfg.num_speakers
        )
        runs = windows_to_turns(
            windows,
            [int(l) for l in labels],
            cell_sec=self.cfg.cell_sec,
            min_turn_sec=self.cfg.min_turn_sec,
        )
        turns = [SpeakerTurn(f"SPEAKER_{lab:02d}", start, end) for lab, start, end in runs]
        self.overlap_regions_ = merge_segments(
            margin_regions(windows, [int(l) for l in labels]) + overlap_regions(turns)
        )
        self.speaker_centroids_ = {}
        if embeddings:
            mat = np.vstack([np.asarray(e, dtype=np.float64) for e in embeddings])
            for lab in sorted({int(l) for l in labels}):
                rows = mat[[i for i, l in enumerate(labels) if int(l) == lab]]
                c = rows.mean(axis=0)
                norm = float(np.linalg.norm(c)) or 1.0
                self.speaker_centroids_[lab] = c / norm
        return merge_turns(turns, gap=self.cfg.hop_sec / 2), embeddings
