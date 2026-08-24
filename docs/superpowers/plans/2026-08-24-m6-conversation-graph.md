# M6 Conversation Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer le chaînage glouton M4 par un graphe d'énoncés dont les arêtes sont scorées par une régression logistique supervisée, regroupées par Louvain, avec commande d'entraînement dédiée et benchmark comparatif contre la baseline M4 conservée.

**Architecture:** Trois modules nouveaux derrière des fonctions pures testables (`pair_features`, `trainer`, `graph_reconstructor`) ; `GraphReconstructor` implémente le même protocole que `HeuristicReconstructor` donc le pipeline reste injectable sans modification ; sélection par `PipelineConfig.reconstructor_kind` ; poids du classifieur persistés en JSON (`models/graph_lr.json`, gitignored).

**Tech Stack:** Python 3.12, uv, scikit-learn (StandardScaler, LogisticRegression, cross_val_score), scipy.special.expit, networkx>=3.2 (Louvain), numpy, typer/rich (CLI), pytest/ruff.

**Spec:** docs/superpowers/specs/2026-08-24-m6-conversation-graph-design.md

## Global Constraints

- Python 3.12, gestion uv (`uv sync` doit suffire sur checkout neuf).
- Torch depuis l'index `https://download.pytorch.org/whl/cu124` (ne pas toucher aux sections `[tool.uv]`).
- Audio canonique interne : mono, 16000 Hz, float32, plage [-1, 1].
- Aucune dépendance gated HF ; aucun appel cloud.
- Pas de commentaires dans le code (style repo) ; docstrings une ligne autorisées.
- Tests unitaires < 10 s sans téléchargement ; intégration sous marqueur `slow`.
- Chaque tâche se termine par un commit vert (`make lint test` passe).
- ruff line-length 95, `ruff format --check` propre.
- Les seeds d'entraînement partent de la base 3000 (jamais 1234… réservés à l'éval).

---

### Task 1: Dépendance networkx + GraphConfig + reconstructor_kind

**Files:**
- Modify: `pyproject.toml` (dépendances)
- Create: rien (config dans fichiers existants)
- Modify: `src/conversation_deconvolution/core/config.py`, `configs/default.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces :
  - `GraphConfig(model_path="models/graph_lr.json", max_gap=30.0, tau=4.0, edge_threshold=0.5, resolution=1.0, seed=0, negative_ratio=3.0)` dataclass.
  - `PipelineConfig.graph: GraphConfig` ; `PipelineConfig.reconstructor_kind: str = "heuristic"`.
  - `PipelineConfig.from_yaml` accepte la section `graph:` et la clé racine `reconstructor_kind:`.

- [ ] **Step 1:** Ajouter `"networkx>=3.2",` à la liste `dependencies` de `pyproject.toml` (après `"matplotlib>=3.9",`) puis lancer `uv sync` (met à jour `uv.lock`).

- [ ] **Step 2:** Écrire le test échouant — ajouter à `tests/unit/test_config.py` :

```python
def test_graph_section_loads(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "graph:\n  edge_threshold: 0.6\n  resolution: 1.2\n"
        "reconstructor_kind: graph\n"
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.graph.edge_threshold == 0.6
    assert cfg.graph.resolution == 1.2
    assert cfg.reconstructor_kind == "graph"


def test_graph_defaults():
    cfg = PipelineConfig.default()
    assert cfg.reconstructor_kind == "heuristic"
    assert cfg.graph.model_path == "models/graph_lr.json"
    assert cfg.graph.negative_ratio == 3.0
```

- [ ] **Step 3:** Run `uv run pytest -q tests/unit/test_config.py` → FAIL (`GraphConfig` inconnue / champ absent).

- [ ] **Step 4:** Dans `core/config.py`, ajouter après `SyntheticConfig` :

```python
@dataclass
class GraphConfig:
    model_path: str = "models/graph_lr.json"
    max_gap: float = 30.0
    tau: float = 4.0
    edge_threshold: float = 0.5
    resolution: float = 1.0
    seed: int = 0
    negative_ratio: float = 3.0
```

Puis dans `PipelineConfig`, ajouter deux champs :

```python
    graph: GraphConfig = field(default_factory=GraphConfig)
    reconstructor_kind: str = "heuristic"
```

Et dans le dict `sections` de `from_yaml` :

```python
            "synthetic": SyntheticConfig,
            "graph": GraphConfig,
```

(`reconstructor_kind` est une clé racine : elle passe déjà par le chemin `else: kwargs[key] = value`.)

- [ ] **Step 5:** Run `uv run pytest -q tests/unit/test_config.py` → PASS.

- [ ] **Step 6:** Ajouter la section à `configs/default.yaml` (après `synthetic:`, avant `text_embedding_model:`) :

```yaml
graph:
  model_path: models/graph_lr.json
  max_gap: 30.0
  tau: 4.0
  edge_threshold: 0.5
  resolution: 1.0
  seed: 0
  negative_ratio: 3.0
reconstructor_kind: heuristic
```

- [ ] **Step 7:** Run `make lint test` → PASS. Commit :

```bash
git add pyproject.toml uv.lock src/conversation_deconvolution/core/config.py configs/default.yaml tests/unit/test_config.py
git commit -m "feat(config): section graph + reconstructor_kind + dependance networkx"
```

### Task 2: Features de paire (`pair_features.py`)

**Files:**
- Create: `src/conversation_deconvolution/conversation/pair_features.py`
- Test: `tests/unit/test_pair_features.py`

**Interfaces:**
- Consumes: `Utterance` (core/types), `gap()` et `alternation()` de `conversation/features.py`.
- Produces :
  - `FEATURE_NAMES: list[str]` = `["gap_sec", "log1p_gap", "temporal_exp", "alternation", "same_speaker", "overlap_ratio", "semantic_cos", "index_distance", "duration_ratio"]` (ordre figé).
  - `pair_feature_names() -> list[str]` (copie défensive).
  - `pair_features(a: Utterance, b: Utterance, rank_a: int, rank_b: int, semantic_cos: float, tau: float) -> list[float]` — même ordre que FEATURE_NAMES.

- [ ] **Step 1:** Écrire le test échouant `tests/unit/test_pair_features.py` :

```python
import math

import pytest

from conversation_deconvolution.conversation.pair_features import (
    pair_feature_names,
    pair_features,
)
from conversation_deconvolution.core.types import Utterance


def U(uid, spk, start, end):
    return Utterance(uid, spk, start, end)


def test_disjoint_pair_features():
    a = U("u1", "A", 0.0, 1.0)
    b = U("u2", "B", 3.0, 4.0)
    v = pair_features(a, b, 0, 5, 0.25, tau=4.0)
    assert v[0] == pytest.approx(2.0)
    assert v[1] == pytest.approx(math.log(3.0))
    assert v[2] == pytest.approx(math.exp(-0.5))
    assert v[3] == 1.0
    assert v[4] == 0.0
    assert v[5] == 0.0
    assert v[6] == 0.25
    assert v[7] == 5.0
    assert v[8] == 1.0


def test_overlapping_pair_clamps_gap_and_ratios_overlap():
    a = U("u1", "A", 0.0, 2.0)
    b = U("u2", "B", 1.0, 3.0)
    v = pair_features(a, b, 0, 1, -0.1, tau=1.0)
    assert v[0] == 0.0
    assert v[1] == 0.0
    assert v[2] == 1.0
    assert v[5] == pytest.approx(0.5)


def test_same_speaker_and_none_speaker():
    a = U("u1", "A", 0.0, 1.0)
    b = U("u2", "A", 2.0, 3.0)
    c = U("u3", None, 2.0, 3.0)
    assert pair_features(a, b, 0, 1, 0.0, 1.0)[4] == 1.0
    assert pair_features(a, b, 0, 1, 0.0, 1.0)[3] == 0.0
    assert pair_features(a, c, 0, 1, 0.0, 1.0)[4] == 0.0


def test_names_match_vector_length():
    a = U("u", "A", 0.0, 1.0)
    b = U("v", "B", 1.0, 2.0)
    assert len(pair_feature_names()) == len(pair_features(a, b, 0, 1, 0.0, 1.0))
```

- [ ] **Step 2:** Run `uv run pytest -q tests/unit/test_pair_features.py` → FAIL (module absent).

- [ ] **Step 3:** Implémenter `src/conversation_deconvolution/conversation/pair_features.py` :

```python
import math

from conversation_deconvolution.conversation.features import alternation, gap
from conversation_deconvolution.core.types import Utterance

FEATURE_NAMES = [
    "gap_sec",
    "log1p_gap",
    "temporal_exp",
    "alternation",
    "same_speaker",
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
    same = 1.0 if (a.speaker is not None and a.speaker == b.speaker) else 0.0
    ratio = (
        min(dur_a, dur_b) / max(dur_a, dur_b)
        if dur_a > 0 and dur_b > 0
        else 0.0
    )
    return [
        g,
        math.log1p(g),
        math.exp(-g / tau),
        alternation(a, b),
        same,
        _overlap_ratio(a, b),
        float(semantic_cos),
        float(rank_b - rank_a),
        ratio,
    ]
```

- [ ] **Step 4:** Run `uv run pytest -q tests/unit/test_pair_features.py` → PASS.

- [ ] **Step 5:** Run `make lint test` → PASS. Commit :

```bash
git add src/conversation_deconvolution/conversation/pair_features.py tests/unit/test_pair_features.py
git commit -m "feat(conversation): features relationnelles de paire pour le graphe m6"
```

### Task 3: Entraîneur LR + persistance JSON (`trainer.py`)

**Files:**
- Create: `src/conversation_deconvolution/conversation/trainer.py`
- Test: `tests/unit/test_trainer.py`

**Interfaces:**
- Produces :
  - `fit_edge_classifier(X: array(n,d), y: array(n), feature_names: list[str], seed: int = 0) -> dict` avec clés `feature_names`, `scaler{mean,scale}`, `coef` (liste d), `intercept` (float), `meta{pairwise_cv_f1}`.
  - `save_model(model: dict, path: str|Path) -> Path`.
  - `load_model(path: str|Path) -> dict` — lève `ValueError` si clés requises manquantes.

- [ ] **Step 1:** Écrire le test échouant `tests/unit/test_trainer.py` :

```python
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
```

- [ ] **Step 2:** Run `uv run pytest -q tests/unit/test_trainer.py` → FAIL (module absent).

- [ ] **Step 3:** Implémenter `src/conversation_deconvolution/conversation/trainer.py` :

```python
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
```

- [ ] **Step 4:** Run `uv run pytest -q tests/unit/test_trainer.py` → PASS.

- [ ] **Step 5:** Run `make lint test` → PASS. Commit :

```bash
git add src/conversation_deconvolution/conversation/trainer.py tests/unit/test_trainer.py
git commit -m "feat(conversation): entraineur lr d aretes + persistance json"
```

### Task 4: Jeu d'entraînement depuis la GT (`build_training_set`)

**Files:**
- Modify: `src/conversation_deconvolution/conversation/trainer.py` (ajout de fonction)
- Test: `tests/unit/test_training_set.py`

**Interfaces:**
- Consumes: `conversation_from_dict` (core/types), `candidate_pairs(utterances, max_gap)` (conversation/features, retourne des indices originaux i<j en ordre chronologique), `pair_features(...)` (Task 2), `GraphConfig` (Task 1).
- Produces: `build_training_set(dataset_dirs: list[Path], embedder, config: GraphConfig, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]` — tous les positifs + négatifs tirés sans remise à hauteur `round(negative_ratio × n_positifs)` ; lève `ValueError` si zéro positif.

- [ ] **Step 1:** Écrire le test échouant `tests/unit/test_training_set.py` :

```python
import json

import numpy as np

from conversation_deconvolution.conversation.trainer import build_training_set
from conversation_deconvolution.core.config import GraphConfig


class TwoTopicEmbedder:
    def encode(self, texts):
        return np.array(
            [[1.0, 0.0] if "cafe" in t else [0.0, 1.0] for t in texts]
        )


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
    assert X1.shape[1] == 9
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
```

(Ne pas oublier `import pytest` en haut du fichier.)

- [ ] **Step 2:** Run `uv run pytest -q tests/unit/test_training_set.py` → FAIL (fonction absente).

- [ ] **Step 3:** Implémenter dans `trainer.py` (imports ajoutés en tête : `from pathlib import Path` déjà présent ; ajouter `from conversation_deconvolution.conversation.features import candidate_pairs` et `from conversation_deconvolution.conversation.pair_features import pair_features` et `from conversation_deconvolution.core.types import conversation_from_dict` et `from conversation_deconvolution.core.config import GraphConfig`) :

```python
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
    n_neg = min(len(negatives), int(round(config.negative_ratio * len(positives))))
    rng = np.random.default_rng(rng_seed)
    picked = rng.choice(len(negatives), size=n_neg, replace=False) if n_neg else []
    kept = positives + [negatives[k] for k in picked]
    X = np.asarray([r[0] for r in kept], dtype=np.float64)
    y = np.asarray([r[1] for r in kept])
    return X, y
```

- [ ] **Step 4:** Run `uv run pytest -q tests/unit/test_training_set.py` → PASS.

- [ ] **Step 5:** Run `make lint test` → PASS. Commit :

```bash
git add src/conversation_deconvolution/conversation/trainer.py tests/unit/test_training_set.py
git commit -m "feat(conversation): jeu d'entrainement d'aretes depuis la verite terrain"
```

### Task 5: Commande CLI `train` + cible Makefile

**Files:**
- Modify: `src/conversation_deconvolution/cli.py`, `Makefile`

**Interfaces:**
- Consumes: Task 3 (`fit_edge_classifier`, `save_model`), Task 4 (`build_training_set`), Task 2 (`pair_feature_names`), `SentenceTransformerEmbedder` (conversation/semantic), `SyntheticGenerator`+`PiperTts` (synthetic).
- Produces: commande `deconvolute train --datasets 8 --out models/graph_lr.json --seed-base 3000 --conversations 2 --speakers 2 [--config]` ; cible `make train`.

- [ ] **Step 1:** Dans `cli.py`, ajouter l'import en tête : `from datetime import UTC, datetime`. Puis ajouter la commande (après `benchmark`) :

```python
@app.command()
def train(
    datasets: int = typer.Option(8, "--datasets"),
    out: Path = typer.Option("models/graph_lr.json", "--out"),
    seed_base: int = typer.Option(3000, "--seed-base"),
    conversations: int = typer.Option(2, "--conversations"),
    speakers: int = typer.Option(2, "--speakers"),
    config_path: Path = typer.Option(None, "--config", "-c"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    from conversation_deconvolution.conversation.pair_features import (
        pair_feature_names,
    )
    from conversation_deconvolution.conversation.semantic import (
        SentenceTransformerEmbedder,
    )
    from conversation_deconvolution.conversation.trainer import (
        build_training_set,
        fit_edge_classifier,
        save_model,
    )
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import PiperTts

    generator = SyntheticGenerator(PiperTts(), cfg.synthetic)
    dirs = []
    for k in range(datasets):
        target = Path(f"data/synthetic/train_{seed_base}_{k}")
        generator.generate(
            target,
            seed=seed_base + k,
            n_conversations=conversations,
            speakers_per_thread=speakers,
        )
        dirs.append(target)
        console.print(f"[green]✓[/green] dataset {k + 1}/{datasets} → {target}")
    embedder = SentenceTransformerEmbedder(cfg.text_embedding_model)
    X, y = build_training_set(dirs, embedder, cfg.graph, rng_seed=cfg.graph.seed)
    model = fit_edge_classifier(X, y, pair_feature_names(), seed=cfg.graph.seed)
    model["meta"].update(
        {
            "n_datasets": datasets,
            "seed_base": seed_base,
            "negative_ratio": cfg.graph.negative_ratio,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    save_model(model, out)
    console.print(
        f"[green]✓[/green] classifieur entraîné "
        f"({len(y)} paires, F1 arêtes CV={model['meta']['pairwise_cv_f1']:.3f}) → {out}"
    )
```

- [ ] **Step 2:** Vérifier la surface CLI sans téléchargement : `uv run deconvolute train --help` → affiche les options sans erreur.

- [ ] **Step 3:** Ajouter au `Makefile` (avant `.PHONY`) et étendre `.PHONY` :

```makefile
train:
	uv run deconvolute train --datasets 8 --out models/graph_lr.json
```

`.PHONY: lint format test test-slow benchmark train`

- [ ] **Step 4:** Run `make lint test` → PASS. Commit :

```bash
git add src/conversation_deconvolution/cli.py Makefile
git commit -m "feat(cli): commande train du classifieur d'aretes + cible make train"
```

### Task 6: `GraphReconstructor`

**Files:**
- Create: `src/conversation_deconvolution/conversation/graph_reconstructor.py`
- Test: `tests/unit/test_graph_reconstructor.py`

**Interfaces:**
- Consumes: `candidate_pairs`, `pair_features`, `load_model` (Tasks 2–3), `GraphConfig` (Task 1).
- Produces: `GraphReconstructor(text_embedder, config: GraphConfig)` avec `reconstruct(utterances: list[Utterance]) -> list[Conversation]` — même protocole que `HeuristicReconstructor(text_embedder, config)`.

- [ ] **Step 1:** Écrire le test échouant `tests/unit/test_graph_reconstructor.py` :

```python
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
    "alternation",
    "same_speaker",
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
        "scaler": {"mean": [0.0] * 9, "scale": [1.0] * 9},
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
    r = GraphReconstructor(
        TwoTopicEmbedder(), GraphConfig(model_path=str(tmp_path / "absent.json"))
    )
    with pytest.raises(FileNotFoundError):
        r.reconstruct(interleaved())
```

Note sur `test_missing_model_file` : selon l'implémentation choisie (chargement au constructeur ou au premier appel), l'erreur peut être levée à la construction — adapter le test pour construire ET appeler sous le même `pytest.raises` si nécessaire :

```python
def test_missing_model_file(tmp_path):
    cfg = GraphConfig(model_path=str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        GraphReconstructor(TwoTopicEmbedder(), cfg).reconstruct(interleaved())
```

- [ ] **Step 2:** Run `uv run pytest -q tests/unit/test_graph_reconstructor.py` → FAIL (module absent).

- [ ] **Step 3:** Implémenter `src/conversation_deconvolution/conversation/graph_reconstructor.py` :

```python
import networkx as nx
import numpy as np
from scipy.special import expit

from conversation_deconvolution.conversation.features import candidate_pairs
from conversation_deconvolution.conversation.pair_features import pair_features
from conversation_deconvolution.conversation.trainer import load_model
from conversation_deconvolution.core.config import GraphConfig
from conversation_deconvolution.core.types import Conversation, Utterance


class GraphReconstructor:
    def __init__(self, text_embedder, config: GraphConfig):
        self.embedder = text_embedder
        self.cfg = config
        self.model = load_model(config.model_path)

    def reconstruct(self, utterances: list[Utterance]) -> list[Conversation]:
        if not utterances:
            return []
        ordered = sorted(utterances, key=lambda u: (u.start, u.end))
        embs = self._normalize(self.embedder.encode([u.text for u in ordered]))
        probs = self._edge_probabilities(ordered, embs)
        communities = self._communities(ordered, probs)
        return self._to_conversations(ordered, communities)

    def _edge_probabilities(self, ordered, embs) -> dict[tuple[int, int], float]:
        pairs = candidate_pairs(ordered, self.cfg.max_gap)
        if not pairs:
            return {}
        X = np.asarray(
            [
                pair_features(
                    ordered[i],
                    ordered[j],
                    i,
                    j,
                    float(np.dot(embs[i], embs[j])),
                    self.cfg.tau,
                )
                for i, j in pairs
            ],
            dtype=np.float64,
        )
        mean = np.asarray(self.model["scaler"]["mean"], dtype=np.float64)
        scale = np.asarray(self.model["scaler"]["scale"], dtype=np.float64)
        coef = np.asarray(self.model["coef"], dtype=np.float64)
        scores = (X - mean) / scale @ coef + float(self.model["intercept"])
        return {(i, j): float(p) for (i, j), p in zip(pairs, expit(scores))}

    def _communities(self, ordered, probs):
        G = nx.Graph()
        G.add_nodes_from(range(len(ordered)))
        for (i, j), p in probs.items():
            if p >= self.cfg.edge_threshold:
                G.add_edge(i, j, weight=p)
        return nx.community.louvain_communities(
            G, weight="weight", resolution=self.cfg.resolution, seed=self.cfg.seed
        )

    def _to_conversations(self, ordered, communities) -> list[Conversation]:
        groups = sorted(communities, key=lambda m: ordered[min(m)].start)
        conversations = []
        for rank, members in enumerate(groups, start=1):
            utts = [ordered[i] for i in sorted(members)]
            participants = list(dict.fromkeys(u.speaker for u in utts if u.speaker))
            conversations.append(
                Conversation(
                    id=f"conversation_{rank:02d}",
                    participants=participants,
                    utterances=utts,
                )
            )
        return conversations

    @staticmethod
    def _normalize(embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
```

- [ ] **Step 4:** Run `uv run pytest -q tests/unit/test_graph_reconstructor.py` → PASS (si Louvain scinde différemment les singletons du test `low_intercept`, vérifier que chaque énoncé est bien seul ; ajuster l'assertion sur le nombre, jamais les membres attendus du test principal).

- [ ] **Step 5:** Run `make lint test` → PASS. Commit :

```bash
git add src/conversation_deconvolution/conversation/graph_reconstructor.py tests/unit/test_graph_reconstructor.py
git commit -m "feat(conversation): reconstructeur graphe lr + louvain"
```

### Task 7: Câblage pipeline + benchmark comparatif

**Files:**
- Modify: `src/conversation_deconvolution/pipeline.py` (extraction `build_reconstructor` + usage dans `build_pipeline`)
- Modify: `src/conversation_deconvolution/cli.py` (`--reconstructor`, refonte `run_benchmark`)
- Test: `tests/unit/test_pipeline_reconstructor.py`

**Interfaces:**
- Consumes: Task 6 (`GraphReconstructor`), `HeuristicReconstructor` existant, `evaluate_results` existant.
- Produces :
  - `build_reconstructor(config: PipelineConfig, text_embedder)` dans `pipeline.py` — `ValueError` si kind inconnu.
  - Option CLI sur `run` : `--reconstructor/-r` (surcharge `cfg.reconstructor_kind` si fournie).
  - `run_benchmark(n_datasets: int, base_seed: int, cfg: PipelineConfig, kinds: list[str]) -> str` — une section `## Reconstruteur : <kind>` par variante, mêmes datasets seedés pour toutes.
  - `format_section(kind: str, rows: list[dict]) -> list[str]` (pure).
  - CLI : `benchmark --reconstructor heuristic|graph|both` (défaut `heuristic`).

- [ ] **Step 1:** Écrire le test échouant `tests/unit/test_pipeline_reconstructor.py` :

```python
import json

import pytest

from conversation_deconvolution.conversation.graph_reconstructor import (
    GraphReconstructor,
)
from conversation_deconvolution.conversation.reconstructor import (
    HeuristicReconstructor,
)
from conversation_deconvolution.cli import format_section
from conversation_deconvolution.core.config import PipelineConfig

NAMES = ["gap_sec", "log1p_gap", "temporal_exp", "alternation", "same_speaker", "overlap_ratio", "semantic_cos", "index_distance", "duration_ratio"]


class NullEmbedder:
    def encode(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4))


def test_build_reconstructor_graph(tmp_path):
    from conversation_deconvolution.pipeline import build_reconstructor

    model = {
        "feature_names": NAMES,
        "scaler": {"mean": [0.0] * 9, "scale": [1.0] * 9},
        "coef": [0.0] * 9,
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
```

- [ ] **Step 2:** Run `uv run pytest -q tests/unit/test_pipeline_reconstructor.py` → FAIL.

- [ ] **Step 3:** Dans `pipeline.py`, extraire la fonction (niveau module, après `DeconvolutionPipeline`) :

```python
def build_reconstructor(config: PipelineConfig, text_embedder):
    if config.reconstructor_kind == "graph":
        from conversation_deconvolution.conversation.graph_reconstructor import (
            GraphReconstructor,
        )

        return GraphReconstructor(text_embedder, config.graph)
    if config.reconstructor_kind == "heuristic":
        from conversation_deconvolution.conversation.reconstructor import (
            HeuristicReconstructor,
        )

        return HeuristicReconstructor(text_embedder, config.reconstruction)
    raise ValueError(f"unknown reconstructor_kind: {config.reconstructor_kind}")
```

Et remplacer dans `build_pipeline` les lignes actuelles :

```python
    reconstructor = HeuristicReconstructor(text_embedder, config.reconstruction)
```

par :

```python
    reconstructor = build_reconstructor(config, text_embedder)
```

(l'import direct de `HeuristicReconstructor` dans `build_pipeline` devient inutile — le supprimer.)

- [ ] **Step 4:** Dans `cli.py` :

Option du command `run` (après `plot`, même style que `--separate`) :

```python
    reconstructor: str = typer.Option(None, "--reconstructor", "-r"),
```

et dans le corps de `run`, après le chargement de la config :

```python
    if reconstructor is not None:
        cfg.reconstructor_kind = reconstructor
```

Option du command `benchmark` :

```python
    reconstructor: str = typer.Option("heuristic", "--reconstructor"),
```

Validation + appel (remplacer `report = run_benchmark(datasets, seed, cfg)`) :

```python
    kinds = {
        "both": ["heuristic", "graph"],
        "heuristic": ["heuristic"],
        "graph": ["graph"],
    }.get(reconstructor)
    if kinds is None:
        raise typer.BadParameter("--reconstructor: heuristic|graph|both")
    report = run_benchmark(datasets, seed, cfg, kinds)
```

Refonte de `run_benchmark` (génère les datasets une seule fois, évalue par kind) :

```python
def run_benchmark(n_datasets: int, base_seed: int, cfg: PipelineConfig, kinds: list[str]) -> str:
    from conversation_deconvolution.audio.loader import load_audio
    from conversation_deconvolution.core.types import result_from_dict
    from conversation_deconvolution.pipeline import build_pipeline
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import PiperTts

    generator = SyntheticGenerator(PiperTts(), cfg.synthetic)
    n_speakers = 2 * 2
    cfg.diarization.num_speakers = n_speakers
    ds_dirs = []
    for k in range(n_datasets):
        ds_dir = Path(f"data/synthetic/bench_{base_seed}_{k}")
        ds_dirs.append(generator.generate(ds_dir, seed=base_seed + k))
    lines = [
        "# Benchmark — Conversation Deconvolution",
        "",
        f"- datasets : {n_datasets}",
        f"- locuteurs (oracle) : {n_speakers}",
        f"- seeds : {base_seed}…{base_seed + n_datasets - 1}",
        "",
    ]
    for kind in kinds:
        cfg.reconstructor_kind = kind
        pipeline = build_pipeline(cfg)
        rows = []
        for gen_dir in ds_dirs:
            result = pipeline.run(load_audio(gen_dir / "mixed.wav"))
            gt_result = result_from_dict(_load(gen_dir / "ground_truth.json"))
            rows.append(evaluate_results(gt_result, result))
        lines += format_section(kind, rows)
        lines.append("")
    return "\n".join(lines) + "\n"


def format_section(kind: str, rows: list[dict]) -> list[str]:
    header = ["DER", "WER (non-overlap)", "pairwise_F1", "ARI", "NMI"]
    if any(r.get("WER (overlap)") is not None for r in rows):
        header.insert(2, "WER (overlap)")
    out = [
        f"## Reconstruteur : {kind}",
        "",
        "| Métrique | moyenne | écart-type |",
        "|---|---|---|",
    ]
    for h in header:
        values = [r[h] for r in rows if r.get(h) is not None]
        if not values:
            out.append(f"| {h} | - | - |")
        else:
            out.append(f"| {h} | {np.mean(values):.4f} | {np.std(values):.4f} |")
    return out
```

- [ ] **Step 5:** Run `uv run pytest -q tests/unit/test_pipeline_reconstructor.py` puis `make lint test` → PASS.

- [ ] **Step 6:** Commit :

```bash
git add src/conversation_deconvolution/pipeline.py src/conversation_deconvolution/cli.py tests/unit/test_pipeline_reconstructor.py
git commit -m "feat(pipeline,cli): reconstructeur graphe brancheable + benchmark comparatif both"
```

### Task 8: Entraînement réel, benchmark comparatif, ADR + ROADMAP + README

**Files:**
- Create: `docs/adr/0009-graphe-supervise-louvain.md`
- Modify: `docs/ROADMAP.md`, `README.md`
- Générés locaux (gitignored) : `models/graph_lr.json`, `reports/benchmark.md`

**Interfaces:**
- Consumes: Tasks 1–7 complets.

- [ ] **Step 1:** Entraîner (réseau + GPU, ~quelques minutes) :

```bash
uv run deconvolute train --datasets 8 --out models/graph_lr.json
```

Attendu : F1 arêtes CV > 0,75 affiché. Si < 0,6, STOP : signaler le chiffre (features probablement non discriminantes, revoir la spec avant d'aller plus loin).

- [ ] **Step 2:** Benchmark comparatif :

```bash
uv run deconvolute benchmark --datasets 4 --seed 1234 --reconstructor both --out reports/benchmark.md
```

Critère d'acceptation (spec) : colonne graph avec **pairwise_F1 ≥ 0,5064 et ARI ≥ 0,2011** (baseline heuristic N3-OFF). Si inférieur → STOP et rapporter les chiffres (itération features/thresholds requise, ne pas cocher M6).

- [ ] **Step 3:** Écrire `docs/adr/0009-graphe-supervise-louvain.md` :

```markdown
# ADR-0009: Reconstruction par graphe supervisé — LR d'arêtes + Louvain

## Status

Accepted

## Context

Le critère de révision de l'ADR-0005 est atteint : pairwise-F1 plafonnée
à ~0,51 malgré les réglages du chaînage glouton par flux. Le projet
dispose d'une vérité terrain synthétique abondante et reproductible,
inexploitée par la baseline heuristique.

## Decision

La reconstruction passe par un graphe explicite : nœuds = énoncés, arêtes
candidates (gap ≤ max_gap) scorées par une régression logistique sur 9
features de paire (gap, log-gap, exp temporelle, alternance,
même-locuteur, taux de chevauchement, cosinus sémantique, distance
d'indice, ratio de durées), entraînée sur des datasets synthétiques seedés
(base 3000, disjointe des seeds d'évaluation 1234). Regroupement par
Louvain sur le graphe pondéré seuillé (résolution et seed configurables).
La baseline M4 (chaînage par flux) reste disponible et comparable via
`benchmark --reconstructor both`.

## Alternatives considered

- Poids manuels non supervisés : plafond identique au scoring manuel actuel.
- MLP sur features : moins interprétable, sur-apprentissage probable sur
  ~20 énoncés/dataset.
- GNN immédiat : coût fort, aucune baseline supervisée intermédiaire pour
  juger son apport.

## Consequences

### Positive

- Signaux combinés optimalement par apprentissage plutôt qu'à la main.
- Chaque poids est inspectable (JSON versionné, coefficients lisibles).
- Comparaison systématique heuristic vs graph sur les mêmes seeds.

### Negative

- Deuxième chemin de reconstruction à maintenir.
- Risque de transfert limité synthétique → réel (à mesurer en M7).

## Reconsideration criteria

Si pairwise-F1 < baseline M4 après itération sur les features, ou si le
transfert réel s'avère mauvais, réévaluer : features prosodiques,
modèle plus expressif, puis GNN (phase 6 avancée).
```

- [ ] **Step 4:** Mettre à jour `docs/ROADMAP.md` :

Remplacer la section M6 (actuellement « outlook ») par :

```markdown
## M6 — Graphe de conversations *(Phase 6)*

**Livrables**

- [x] Features de paire enrichies (position, durées, chevauchement continu).
- [x] Probabilité d'arête apprise (LR supervisée, seeds train 3000+ disjointes).
- [x] Graphe networkx pondéré + Louvain (seed et résolution configurables).
- [x] `deconvolute train` + persistance JSON des poids.
- [x] Benchmark comparatif `--reconstructor both` vs baseline M4.
- [x] Critère d'acceptation atteint (voir reports/benchmark.md, ADR-0009).

**Acceptation :** pairwise-F1 et ARI ≥ baseline M4 sur le benchmark seedé ;
GNN explicitement hors périmètre de cette itération.
```

Mettre à jour la ligne M6 du tableau État :

```markdown
| M6 | ✅ fait — graphe supervisé (LR 9 features + Louvain) ; chiffres et comparaison M4 dans reports/benchmark.md, décision ADR-0009 |
```

Et la ligne « Prochaine étape » du bloc État devient :

```markdown
**Prochaine étape :** M7 — robustesse (bruit réaliste, réverbération RIR,
interruptions intra-conversation) ; mesure du transfert synthétique → réel
du reconstructeur graphe (ADR-0009).
```

- [ ] **Step 5:** Mettre à jour `README.md` :

Quickstart — ajouter entre `deconvolute synth` et `deconvolute run` :

```bash
# entraîner le classifieur d'arêtes du graphe (seeds train 3000+, ~minutes)
uv run deconvolute train --datasets 8

# pipeline complet avec le reconstructeur graphe
uv run deconvolute run data/synthetic/sample/mixed.wav \
    -o out.json -n 4 -r graph
```

Section Statut :

```markdown
**Phase actuelle :** Phases 0→6 implémentées ; M6 = graphe supervisé
(LR d'arêtes + Louvain, ADR-0009), baseline heuristique conservée et
comparable (`benchmark --reconstructor both`).

**Prochaine étape :** phase 7 — robustesse (bruit, réverbération,
interruptions) et mesure du transfert synthétique → réel.
```

- [ ] **Step 6:** Run `make lint test` → PASS.

- [ ] **Step 7:** Commits :

```bash
git add docs/adr/0009-graphe-supervise-louvain.md docs/ROADMAP.md README.md
git commit -m "docs: adr-0009 graphe supervise + roadmap m6 + quickstart train"
```

## Self-Review

- Couverture spec : §1 pair_features ✓ (Task 2) · §2 GraphReconstructor ✓ (Task 6) · §3 trainer fit/save/load + build_training_set ✓ (Tasks 3–4, meta complétée par CLI Task 5) · §4 GraphConfig + reconstructor_kind + yaml ✓ (Task 1) · §5 CLI train/benchmark/run ✓ (Tasks 5, 7) · §6 networkx ✓ (Task 1) · gestion d'erreurs (modèle absent, zéro positif, kind inconnu) ✓ (Tasks 3, 4, 6, 7) · tests ✓ partout + critères d'acceptation Task 8 · docs ADR/ROADMAP/README ✓ (Task 8).
- Placeholders : aucun TBD ; chaque étape de code contient le code complet.
- Cohérence types : `pair_features(a,b,rank_a,rank_b,semantic_cos,tau)` identique Tasks 2/4/6 · `GraphConfig` champs identiques Tasks 1/4/6/7 · `build_training_set(dataset_dirs, embedder, config, rng_seed)` identique Tasks 4/5 · `format_section(kind, rows) -> list[str]` identique Tasks 7 (impl + tests) · `build_reconstructor(config, text_embedder)` identique Task 7.
