from dataclasses import dataclass

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _span(x) -> tuple[float, float]:
    return (x.start, x.end)


def iou(a, b) -> float:
    sa, ea = _span(a)
    sb, eb = _span(b)
    inter = max(0.0, min(ea, eb) - max(sa, sb))
    union = (ea - sa) + (eb - sb) - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(gt: list, pred: list, min_iou: float = 0.3) -> list[tuple]:
    if not gt or not pred:
        return []
    scores = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            scores[i, j] = iou(g, p)
    rows, cols = linear_sum_assignment(-scores)
    return [
        (gt[r], pred[c])
        for r, c in zip(rows, cols)
        if scores[r, c] >= min_iou
    ]


def wer_report(pairs: list[tuple[str, str]]) -> dict:
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    out = jiwer.process_words(
        refs, hyps,
        reference_transform=jiwer.wer_default,
        hypothesis_transform=jiwer.wer_default,
    )
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "n_words": out.hits + out.substitutions + out.deletions,
    }
