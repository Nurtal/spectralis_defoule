import json

import numpy as np
import pytest

from conversation_deconvolution.conversation.graph_reconstructor import (
    GraphReconstructor,
)
from conversation_deconvolution.core.config import GraphConfig
from conversation_deconvolution.core.types import Utterance

NAMES = [
    "gap_sec",
    "log1p_gap",
    "temporal_exp",
    "overlap_ratio",
    "semantic_cos",
    "index_distance",
    "duration_ratio",
]


class TwoTopicEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if "cafe" in t else [0.0, 1.0] for t in texts])


def write_model(path, coef_by_name, intercept):
    model = {
        "feature_names": NAMES,
        "scaler": {"mean": [0.0] * 7, "scale": [1.0] * 7},
        "coef": [coef_by_name.get(n, 0.0) for n in NAMES],
        "intercept": intercept,
        "meta": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model))
    return path


def interleaved():
    def U(uid, spk, s, e, txt):
        return Utterance(uid, spk, s, e, txt)

    return [
        U("a1", "A", 0.0, 1.8, "tu viens au cafe demain midi"),
        U("c1", "C", 0.9, 2.4, "le rapport final est termine"),
        U("a2", "A", 2.6, 4.0, "parfait pour le cafe alors"),
        U("c2", "D", 3.0, 4.5, "merci pour le rapport beaucoup"),
        U("a3", "B", 4.4, 5.8, "je reponds au cafe plus tard"),
        U("c3", "C", 4.9, 6.4, "le rapport part au courrier"),
    ]


def semantic_model(tmp_path):
    path = write_model(tmp_path / "model.json", {"semantic_cos": 12.0}, -6.0)
    return GraphReconstructor(TwoTopicEmbedder(), GraphConfig(model_path=str(path)))


def test_interleaved_threads_separated(tmp_path):
    convs = semantic_model(tmp_path).reconstruct(interleaved())
    members = sorted(tuple(sorted(u.id for u in c.utterances)) for c in convs)
    assert members == [("a1", "a2", "a3"), ("c1", "c2", "c3")]


def test_ids_and_participants(tmp_path):
    convs = semantic_model(tmp_path).reconstruct(interleaved())
    ids = sorted(c.id for c in convs)
    assert ids == ["conversation_01", "conversation_02"]
    first = next(c for c in convs if "a1" in [u.id for u in c.utterances])
    assert first.participants == ["A", "B"]


def test_deterministic(tmp_path):
    r = semantic_model(tmp_path)
    assert r.reconstruct(interleaved()) == r.reconstruct(interleaved())


def test_low_intercept_gives_singletons(tmp_path):
    path = write_model(tmp_path / "model.json", {}, -20.0)
    r = GraphReconstructor(TwoTopicEmbedder(), GraphConfig(model_path=str(path)))
    convs = r.reconstruct(interleaved())
    assert len(convs) == 6


def test_empty_input(tmp_path):
    assert semantic_model(tmp_path).reconstruct([]) == []


def test_missing_model_file(tmp_path):
    cfg = GraphConfig(model_path=str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        GraphReconstructor(TwoTopicEmbedder(), cfg)
