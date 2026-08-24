# Spécification — M6 Graphe de conversations

Date : 2026-08-24 · Statut : approuvée (brainstorming) · Précède : plan d'implémentation M6

## Objectif

Remplacer le chaînage glouton de la reconstruction M4 par une représentation
explicite en graphe : nœuds = énoncés, arêtes scorées par un classifieur
supervisé, clusters = communautés Louvain. La baseline M4 reste disponible
et comparée sur le même harnais seedé. Déclencheur : critère de révision de
l'ADR-0005 atteint (pairwise-F1 plafonné à 0,51 malgré réglages).

## Décisions structurantes (issues du brainstorming)

1. **Score d'arête** : régression logistique supervisée sur features de
   paire, entraînée sur la vérité terrain synthétique.
2. **Features** : signaux M4 (gap, alternance, sémantique, même-locuteur,
   chevauchement) + position (distance d'indice, ratio de durées).
   Pas de prosodie dans cette itération (dépendance + gain incertain).
3. **Regroupement** : Louvain (`networkx`) sur graphe pondéré seuillé ;
   résolution et seed configurables.
4. **Entraînement** : commande dédiée `deconvolute train`, datasets seedés
   dédiés (base 3000, disjointe des seeds d'évaluation 1234) → aucune fuite
   train/test ; poids persistés en JSON versionné.

## Architecture

### 1. `conversation/pair_features.py` — extraction pure

Fonctions :

- `pair_feature_names() -> list[str]`
- `pair_features(a: Utterance, b: Utterance, rank_a: int, rank_b: int,
  semantic_cos: float, tau: float) -> list[float]`
- `candidate_pairs(utterances, max_gap)` existe déjà dans
  `conversation/features.py` ; réutilisé tel quel.

Vecteur de features (ordre figé, documenté par `pair_feature_names`) :

| # | Nom | Définition |
|---|---|---|
| 1 | `gap_sec` | `max(0, b.start − a.end)` |
| 2 | `log1p_gap` | `log1p(gap_sec)` |
| 3 | `temporal_exp` | `exp(−gap_sec / tau)` |
| 4 | `alternation` | 1 si locuteurs différents (tous deux non nuls) sinon 0 |
| 5 | `same_speaker` | 1 si même locuteur non nul sinon 0 |
| 6 | `overlap_ratio` | `overlap(a,b) / min(dur_a, dur_b)`, clampé [0,1], 0 si pas de recouvrement |
| 7 | `semantic_cos` | cosinus embeddings textes L2-normalisés (précalculé) |
| 8 | `index_distance` | `rank_b − rank_a` (ordre chronologique) |
| 9 | `duration_ratio` | `min(dur_a, dur_b) / max(dur_a, dur_b)` |

Le can't-link dur de M4 (`max_speaker_overlap_ratio`) devient ici une
feature continue (`overlap_ratio`) que le classifieur pondère lui-même.

### 2. `conversation/graph_reconstructor.py` — `GraphReconstructor`

Signature identique à `HeuristicReconstructor` : constructeur
`(text_embedder, config)` + `reconstruct(utterances) -> list[Conversation]`.
Interchangeable dans `DeconvolutionPipeline` sans modification du pipeline.

Chaîne de traitement :

1. Tri chronologique `(start, end)` ; embeddings texte L2-normalisés.
2. Paires candidates via `candidate_pairs(utterances, max_gap)`.
3. Matrice X des features (§1) ; probabilité
   `p = σ(scale(X) · coef + intercept)` avec scaler et poids chargés du JSON.
4. Graphe non orienté : arête (i,j) si `p ≥ edge_threshold`, poids = p.
5. `nx.community.louvain_communities(G, weight="weight",
   resolution=config.resolution, seed=config.seed)`.
6. Conversations triées par première apparition, ids `conversation_%02d`,
   participants par ordre d'apparition — mêmes conventions que M4.

Cas limites : liste vide → `[]` ; graphe sans arête valide → communautés
singletons (comportement Louvain natif) ; fichier de modèle absent ou
invalide → erreur explicite dès le constructeur.

### 3. `conversation/trainer.py` — entraînement

Deux unités séparées :

- `fit_edge_classifier(X, y, seed) -> dict` **pur** (aucune I/O, testable
  sans TTS) : StandardScaler + LogisticRegression(`class_weight="balanced"`,
  `max_iter=1000`, `random_state=seed`) → dictionnaire de poids.
- `build_training_set(generator, cfg, n_datasets, seed_base) -> (X, y,
  métadonnées)` : génère les datasets seedés, lit la GT, construit les
  paires candidates et le label `y = 1` si même id de conversation.
  Équilibrage : tous les positifs conservés ; négatifs tirés sans remise
  (rng seedé) à hauteur de `negative_ratio × nb_positifs`.

Persistance (`models/graph_lr.json`) :

```json
{
  "feature_names": ["gap_sec", "..."],
  "scaler": {"mean": [...], "scale": [...]},
  "coef": [...],
  "intercept": 0.0,
  "meta": {
    "n_datasets": 8, "seed_base": 3000,
    "negative_ratio": 3.0, "trained_at": "2026-08-24T…",
    "pairwise_cv_f1": 0.87
  }
}
```

`pairwise_cv_f1` : F1 d'appariement arête calculée en validation croisée
5 folds pendant l'entraînement (diagnostic sans fuite, stocké en méta).

### 4. Config (`core/config.py` + `configs/default.yaml`)

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

- `PipelineConfig.graph: GraphConfig` (section yaml `graph:`).
- `PipelineConfig.reconstructor_kind: str = "heuristic"` (`"heuristic"` |
  `"graph"`) ; `build_pipeline` instancie en conséquence.

### 5. CLI (`cli.py`)

- Nouvelle commande :
  `deconvolute train --datasets 8 --out models/graph_lr.json --seed-base 3000
  --conversations 2 --speakers 2 [--config]`.
- `benchmark --reconstructor heuristic|graph|both` (défaut `heuristic`) ;
  `both` produit un tableau par variante, mêmes seeds → comparaison directe.
- `run` honore `reconstructor_kind` via la config existante.

### 6. Dépendance

`networkx>=3.2` ajoutée à `pyproject.toml` (pure Python, aucun binaire).

## Flux de données

```
GT synthétique ──► build_training_set ──► fit_edge_classifier ──► models/graph_lr.json
                                                                          │
audio ─► pipeline ─► utterances ─► GraphReconstructor ◄───────────────────┘
                          │              │ candidats → features → p(edge)
                          │              └─► Louvain ─► conversations
                          ▼
                   TranscriptResult ─► évaluation (harnais inchangé)
```

## Gestion d'erreurs

- Modèle absent/corrompu : `FileNotFoundError`/`ValueError` explicites avec
  chemin et cause, levés au constructeur (fail fast avant tout calcul).
- Énoncés sans locuteur (`speaker=None`) : alternance/same_speaker à 0,
  `overlap_ratio` calculé quand même (indépendant du locuteur).
- Dataset GT vide après filtrage : jeu d'entraînement rejeté avec message.

## Tests

Unitaires purs (< 10 s, sans téléchargement) :

- `test_pair_features.py` : valeurs exactes sur énoncés jouets (clamp du
  gap, alternance, overlap_ratio, distance d'indice, ratio de durées).
- `test_graph_reconstructor.py` : poids LR factices écrits en tmp → deux
  dialogues entrelacés reconstruits en 2 groupes (même scénario que
  `test_reconstructor.py`) ; déterminisme (deux appels ⇒ sortie égale) ;
  erreur claire si fichier de poids manquant.
- `test_trainer.py` : `fit_edge_classifier` sur matrice séparable →
  coefficients déterministes et F1 train élevée ; aller-retour
  sauvegarde/chargement du JSON.

Intégration (marqueur `slow`) : `deconvolute train` réel (TTS Piper) puis
`benchmark --reconstructor both` sur GPU.

## Critères d'acceptation

1. `make lint test` verts ; nouveaux tests couvrant les trois modules.
2. Sur le benchmark seedé (1234…, 4 datasets) :
   **pairwise-F1 ≥ 0,51 et ARI ≥ 0,20** (baselines M4 actuelles), cible
   strictement supérieure ; sinon itération sur les features avant toute
   extension (GNN explicitement hors périmètre de cette itération).
3. Reproductibilité : `deconvolute train` deux fois avec mêmes seeds ⇒ JSON
   identique (hors horodatage) ; benchmark déterministe comme aujourd'hui.

## Docs associées (en fin d'implémentation)

- ADR-0009 : classifieur supervisé + Louvain en remplacement du chaînage
  glouton ; M4 conservé comme baseline ; critères de révision vers GNN.
- ROADMAP : cases M6 cochées + note d'état.
- README : quickstart complété (`deconvolute train`, benchmark comparatif).
