import json

import pytest

from conversation_deconvolution.cli import format_section
from conversation_deconvolution.conversation.graph_reconstructor import (
    GraphReconstructor,
)
from conversation_deconvolution.conversation.reconstructor import (
    HeuristicReconstructor,
)
from conversation_deconvolution.core.config import PipelineConfig

NAMES = [
    "gap_sec",
    "log1p_gap",
    "temporal_exp",
    "overlap_ratio",
    "semantic_cos",
    "index_distance",
    "duration_ratio",
]


class NullEmbedder:
    def encode(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4))


def test_build_reconstructor_graph(tmp_path):
    from conversation_deconvolution.pipeline import build_reconstructor

    model = {
        "feature_names": NAMES,
        "scaler": {"mean": [0.0] * 7, "scale": [1.0] * 7},
        "coef": [0.0] * 7,
        "intercept": 0.0,
        "meta": {},
    }
    p = tmp_path / "model.json"
    p.write_text(json.dumps(model))
    cfg = PipelineConfig()
    cfg.reconstructor_kind = "graph"
    cfg.graph.model_path = str(p)
    r = build_reconstructor(cfg, NullEmbedder())
    assert isinstance(r, GraphReconstructor)


def test_build_reconstructor_heuristic_default():
    from conversation_deconvolution.pipeline import build_reconstructor

    cfg = PipelineConfig()
    assert isinstance(build_reconstructor(cfg, NullEmbedder()), HeuristicReconstructor)


def test_build_reconstructor_unknown_kind():
    from conversation_deconvolution.pipeline import build_reconstructor

    cfg = PipelineConfig()
    cfg.reconstructor_kind = "magic"
    with pytest.raises(ValueError):
        build_reconstructor(cfg, NullEmbedder())


def test_format_section_table():
    rows = [
        {
            "DER": 0.1,
            "WER (non-overlap)": 0.5,
            "WER (overlap)": None,
            "pairwise_F1": 0.6,
            "ARI": 0.2,
            "NMI": 0.3,
        }
    ]
    lines = format_section("graph", rows)
    text = "\n".join(lines)
    assert "## Reconstruteur : graph" in text
    assert "| DER | 0.1000 | 0.0000 |" in text
    assert "WER (overlap)" not in text


def test_format_section_with_overlap_column():
    rows = [
        {
            "DER": 0.1,
            "WER (non-overlap)": 0.5,
            "WER (overlap)": 0.8,
            "pairwise_F1": 0.6,
            "ARI": 0.2,
            "NMI": 0.3,
        },
        {
            "DER": 0.3,
            "WER (non-overlap)": 0.7,
            "WER (overlap)": None,
            "pairwise_F1": 0.4,
            "ARI": 0.1,
            "NMI": 0.2,
        },
    ]
    text = "\n".join(format_section("heuristic", rows))
    assert "| WER (overlap) | 0.8000 | 0.0000 |" in text
