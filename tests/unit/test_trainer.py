import pytest

from conversation_deconvolution.conversation.trainer import (
    fit_edge_classifier,
    load_model,
    save_model,
)


def _separable():
    import numpy as np

    rng = np.random.default_rng(0)
    pos = rng.normal([3.0, -2.0], 0.1, size=(40, 2))
    neg = rng.normal([-3.0, 2.0], 0.1, size=(120, 2))
    X = np.vstack([pos, neg])
    y = np.array([1] * 40 + [0] * 120)
    return X, y


def test_fit_separable_deterministic_and_good(tmp_path):
    X, y = _separable()
    m1 = fit_edge_classifier(X, y, ["f0", "f1"], seed=0)
    m2 = fit_edge_classifier(X, y, ["f0", "f1"], seed=0)
    assert m1 == m2
    assert m1["meta"]["pairwise_cv_f1"] > 0.95
    assert len(m1["coef"]) == 2
    path = save_model(m1, tmp_path / "model.json")
    assert load_model(path) == m1


def test_load_invalid_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"foo": 1}')
    with pytest.raises(ValueError):
        load_model(p)
