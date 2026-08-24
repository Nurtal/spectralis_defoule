import json

import numpy as np
import pytest

from conversation_deconvolution.conversation.trainer import build_training_set
from conversation_deconvolution.core.config import GraphConfig


class TwoTopicEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if "cafe" in t else [0.0, 1.0] for t in texts])


def write_gt(directory, conversations):
    directory.mkdir(parents=True)
    payload = {"conversations": []}
    for cid, utts in conversations:
        payload["conversations"].append(
            {
                "id": cid,
                "participants": [],
                "utterances": [
                    {
                        "id": uid,
                        "speaker": spk,
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                    for uid, spk, start, end, text in utts
                ],
            }
        )
    (directory / "ground_truth.json").write_text(json.dumps(payload))


def test_labels_sampling_determinism(tmp_path):
    d = tmp_path / "ds"
    write_gt(
        d,
        [
            (
                "conversation_01",
                [
                    ("a1", "A", 0.0, 1.0, "au cafe demain"),
                    ("a2", "B", 1.5, 2.5, "le cafe est bon"),
                    ("a3", "A", 3.0, 4.0, "cafe encore"),
                ],
            ),
            (
                "conversation_02",
                [("c1", "C", 0.5, 1.5, "rapport"), ("c2", "D", 2.0, 3.0, "rapport")],
            ),
        ],
    )
    cfg = GraphConfig(max_gap=30.0, tau=4.0, negative_ratio=1.5)
    X1, y1 = build_training_set([d], TwoTopicEmbedder(), cfg, rng_seed=0)
    X2, y2 = build_training_set([d], TwoTopicEmbedder(), cfg, rng_seed=0)
    n_pos = int((y1 == 1).sum())
    n_neg = int((y1 == 0).sum())
    assert n_pos == 4
    assert n_neg == 6
    assert len(X1) == len(y1) == 10
    assert X1.shape[1] == 7
    assert np.array_equal(y1, y2) and np.array_equal(X1, X2)


def test_no_positive_raises(tmp_path):
    d = tmp_path / "ds"
    write_gt(
        d,
        [
            ("conversation_01", [("a1", "A", 0.0, 1.0, "cafe")]),
            ("conversation_02", [("c1", "C", 0.2, 1.2, "rapport")]),
        ],
    )
    with pytest.raises(ValueError):
        build_training_set([d], TwoTopicEmbedder(), GraphConfig(), rng_seed=0)
