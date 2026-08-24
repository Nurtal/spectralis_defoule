from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from conversation_deconvolution.core.types import SpeakerTurn


@dataclass
class DerResult:
    total: float
    correct: float
    miss: float
    false_alarm: float
    confusion: float

    @property
    def der(self) -> float:
        if self.total <= 0:
            return 0.0
        return (self.miss + self.false_alarm + self.confusion) / self.total


def _grid(start: float, end: float, step: float) -> np.ndarray:
    n = max(1, int(round((end - start) / step)))
    return start + (np.arange(n) + 0.5) * step


def _label_frames(frames: np.ndarray, turns: list[SpeakerTurn], shrink: float = 0.0):
    labels = np.full(len(frames), -1, dtype=int)
    for idx, t in enumerate(turns):
        s, e = t.start + shrink, t.end - shrink
        if e <= s:
            continue
        labels[(frames >= s) & (frames < e)] = idx
    return labels


def diarization_error_rate(
    reference: list[SpeakerTurn],
    hypothesis: list[SpeakerTurn],
    collar: float = 0.25,
    step: float = 0.01,
) -> DerResult:
    if not reference:
        return DerResult(0.0, 0.0, 0.0, 0.0, 0.0)
    lo = max(0.0, min([t.start for t in reference] + [t.start for t in hypothesis]) - collar)
    hi = max([t.end for t in reference] + [t.end for t in hypothesis])
    frames = _grid(lo, hi, step)
    ref_labels = _label_frames(frames, reference, shrink=collar)
    hyp_labels = _label_frames(frames, hypothesis)
    ignore = np.zeros(len(frames), dtype=bool)
    for t in reference:
        ignore |= (frames >= t.start - collar) & (frames < t.start + collar)
        ignore |= (frames >= t.end - collar) & (frames < t.end + collar)
    hyp_labels[ignore] = -2

    n_ref = len(reference)
    n_hyp = len(hypothesis)
    count = np.zeros((n_ref, n_hyp))
    for r in range(n_ref):
        mask_r = ref_labels == r
        if not mask_r.any():
            continue
        for h in range(n_hyp):
            count[r, h] = np.sum(mask_r & (hyp_labels == h)) * step

    mapping: dict[int, int] = {}
    if n_hyp:
        rows, cols = linear_sum_assignment(-count)
        mapping = {int(r): int(c) for r, c in zip(rows, cols)}

    correct = miss = fa = conf = 0.0
    for f in range(len(frames)):
        r = ref_labels[f]
        h = hyp_labels[f]
        if h == -2:
            continue
        if r >= 0:
            if h < 0:
                miss += step
            elif mapping.get(r) == h:
                correct += step
            else:
                conf += step
        elif h >= 0:
            fa += step
    total = float(np.sum(ref_labels >= 0) * step)
    return DerResult(total, correct, miss, fa, conf)
