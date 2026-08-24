import math

from conversation_deconvolution.conversation.features import gap
from conversation_deconvolution.core.types import Utterance

FEATURE_NAMES = [
    "gap_sec",
    "log1p_gap",
    "temporal_exp",
    "overlap_ratio",
    "semantic_cos",
    "index_distance",
    "duration_ratio",
]


def pair_feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def _overlap_ratio(a: Utterance, b: Utterance) -> float:
    ov = min(a.end, b.end) - max(a.start, b.start)
    if ov <= 0:
        return 0.0
    shorter = min(a.end - a.start, b.end - b.start)
    if shorter <= 0:
        return 0.0
    return float(min(1.0, ov / shorter))


def pair_features(
    a: Utterance,
    b: Utterance,
    rank_a: int,
    rank_b: int,
    semantic_cos: float,
    tau: float,
) -> list[float]:
    g = gap(a, b)
    dur_a = a.end - a.start
    dur_b = b.end - b.start
    ratio = min(dur_a, dur_b) / max(dur_a, dur_b) if dur_a > 0 and dur_b > 0 else 0.0
    return [
        g,
        math.log1p(g),
        math.exp(-g / tau),
        _overlap_ratio(a, b),
        float(semantic_cos),
        float(rank_b - rank_a),
        ratio,
    ]
