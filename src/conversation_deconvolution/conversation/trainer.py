import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
