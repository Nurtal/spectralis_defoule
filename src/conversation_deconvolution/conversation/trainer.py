import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from conversation_deconvolution.conversation.features import candidate_pairs
from conversation_deconvolution.conversation.pair_features import pair_features
from conversation_deconvolution.core.config import GraphConfig
from conversation_deconvolution.core.types import conversation_from_dict

REQUIRED_KEYS = {"feature_names", "scaler", "coef", "intercept"}


def _lr(seed: int) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)


def fit_edge_classifier(X, y, feature_names: list[str], seed: int = 0) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lr = _lr(seed).fit(Xs, y)
    cv_f1 = cross_val_score(_lr(seed), Xs, y, cv=5, scoring="f1").mean()
    return {
        "feature_names": list(feature_names),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "meta": {"pairwise_cv_f1": float(cv_f1)},
    }


def save_model(model: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=1))
    return path


def load_model(path: str | Path) -> dict:
    model = json.loads(Path(path).read_text())
    missing = REQUIRED_KEYS - set(model)
    if missing:
        raise ValueError(f"invalid model file {path}: missing {sorted(missing)}")
    return model


def build_training_set(dataset_dirs, embedder, config: GraphConfig, rng_seed: int = 0):
    rows: list[tuple[list[float], int]] = []
    for entry in dataset_dirs:
        entry = Path(entry)
        data = json.loads((entry / "ground_truth.json").read_text())
        convs = [conversation_from_dict(c) for c in data["conversations"]]
        utts = sorted(
            (u for c in convs for u in c.utterances),
            key=lambda u: (u.start, u.end),
        )
        conv_of = {u.id: c.id for c in convs for u in c.utterances}
        embs = np.asarray(embedder.encode([u.text for u in utts]), dtype=np.float64)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        for i, j in candidate_pairs(utts, config.max_gap):
            cos = float(np.dot(embs[i], embs[j]))
            x = pair_features(utts[i], utts[j], i, j, cos, config.tau)
            y = 1 if conv_of[utts[i].id] == conv_of[utts[j].id] else 0
            rows.append((x, y))
    positives = [r for r in rows if r[1] == 1]
    negatives = [r for r in rows if r[1] == 0]
    if not positives:
        raise ValueError(f"no positive pairs in training set: {list(dataset_dirs)}")
    n_neg = min(len(negatives), round(config.negative_ratio * len(positives)))
    rng = np.random.default_rng(rng_seed)
    picked = rng.choice(len(negatives), size=n_neg, replace=False) if n_neg else []
    kept = positives + [negatives[k] for k in picked]
    X = np.asarray([r[0] for r in kept], dtype=np.float64)
    y = np.asarray([r[1] for r in kept])
    return X, y
