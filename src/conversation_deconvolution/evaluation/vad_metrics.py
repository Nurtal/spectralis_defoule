import numpy as np

from conversation_deconvolution.core.types import Segment


def frame_flags(probs: np.ndarray, threshold: float) -> np.ndarray:
    return probs >= threshold


def _gt_flags(n_frames: int, frame_rate: float, segments: list[Segment]) -> np.ndarray:
    flags = np.zeros(n_frames, dtype=bool)
    centers = (np.arange(n_frames) + 0.5) / frame_rate
    for seg in segments:
        flags |= (centers >= seg.start) & (centers < seg.end)
    return flags


def vad_prf(
    pred_probs: np.ndarray,
    frame_rate: float,
    gt_segments: list[Segment],
    threshold: float = 0.5,
) -> dict[str, float]:
    pred = frame_flags(np.asarray(pred_probs), threshold)
    gt = _gt_flags(len(pred), frame_rate, gt_segments)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
