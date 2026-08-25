# ADR-0009: Graphe de conversations supervisé (LR d'arêtes + Louvain) rejeté comme reconstruction par défaut

## Status

Rejected (le code est conservé derrière `--reconstructor graph`, la baseline
heuristique M4 reste le comportement standard ; complète ADR-0005)

## Context

Le jalon M6 visait à remplacer le chaînage glouton heuristique par un
graphe d'énoncés : arêtes candidates dans une fenêtre temporelle, features
de paire, régression logistique calibrée sur données synthétiques étiquetées,
probabilité d'arête σ(scale·coef + intercept), partition par Louvain.
L'infrastructure a été livrée (features de paire ordonnées, entraîneur avec
persistance JSON, commande `train` seedée 3000+ disjointe des seeds
d'évaluation 1234+, `GraphReconstructor`, sélection par
`PipelineConfig.reconstructor_kind`, benchmark comparatif `--reconstructor
both`). Critère d'acceptation spec : battre la baseline heuristic sur le
benchmark seedé — pairwise-F1 ≥ 0,5064 et ARI ≥ 0,2011.

Chaîne de preuves (4 datasets seedés 1234…1237, num_speakers oracle) :

| Variante | F1 arêtes CV | pairwise-F1 bench | ARI bench |
|---|---|---|---|
| heuristic (baseline) | — | **0.5064 → 0.5000**¹ | **0.2011 → 0.2577**¹ |
| graph v1 : 9 features dont identité locuteur | 0.786 | 0.4670 | 0.0095 |
| graph v2 : 7 features sans identité | 0.650 | ≤ v1 tous réglages | ≈ 0 |
| graph v2 après front-end amélioré ¹ | 0.650 | 0.4390 | −0.0515 |

¹ après itération front-end (voir ci-dessous) ; la comparaison vaut au sein
d'un même run.

Diagnostics établis en route :

- **Raccourci d'identité (v1)** : `same_speaker` prédit parfaitement les
  paires positives à l'entraînement (pools de voix disjoints par
  conversation) mais produit à l'éval des communautés = clusters de voix,
  or une conversation contient ses deux voix → ARI ≈ 0. Retrait des
  features d'identité (v2) confirmé nécessaire mais insuffisant.
- **Balayages exhaustifs sans gain** : seuil d'arête × résolution Louvain
  (11 × 5 points, condition diarization corrigée), puis agrégation
  voix-niveau (max/mean des probabilités croisées, partition connexe des
  voix) : aucune combinaison n'approche le critère ; dès τ ≥ 0,30 toutes
  les paires de voix sont connectées.
- **Signal sémantique structurellement faible** : même sur textes propres
  d'entraînement, cosinus croix-voix meme-conversation vs autres =
  0.370 vs 0.331 (écart ~0,04).
- **Itération front-end (pad ASR)** : correction d'un artefact métrique
  (WER non-overlap = défaut 1.0 sur ensemble vide — quasi aucun énoncé GT
  sans chevauchement dans ce layout entrelacé) et adoption de
  `context_pad_sec: 0` (WER −6,6 pts reproductible, ARI heuristic
  +0,057). Mais au niveau des sorties pipeline : cosinus anti-corrélé
  (pos < nég sur 2 datasets), écarts temporels inversés ou bruités, et
  diarization fusionnant les voix inter-conversations sur 2 datasets sur
  4. L'information « conversation » n'est plus présente en sortie du
  front-end pour des features de paire texte + temps.

## Decision

L'approche graphe supervisé texte + temps est **rejetée comme
remplacement** de la reconstruction heuristique sur ce benchmark : elle ne
peut pas extraire une information détruite en amont. Le code reste intégré
et activable (`--reconstructor graph`, modèle entraînable via
`deconvolute train` / `make train`) : l'infrastructure (features, LR,
Louvain, benchmark comparatif) est saine et réutilisable. La baseline
heuristique M4 demeure le reconstructeur par défaut.

## Alternatives considered

- Itérer sur les features prosodiques/position (spéc M6 initiale) :
  rejeté en l'état, le goulot est l'amont (contamination et fusion de
  voix), pas la richesse des features de paire.
- Entraînement en conditions bruitées (textes ASR simulés) : insuffisant
  seul, le cosinus étant anti-corrélé à l'éval ; reporté dans une approche
  « graphe de voix » plus large (co-occurrence temporelle entre voix),
  candidate pour une itération future.
- Ajuster le critère d'acceptation : rejeté, cela trahirait la spec M6.
- Supprimer le code : rejeté, coût de maintenance faible derrière le flag,
  benchmark comparatif utile pour toute itération amont future.

## Consequences

### Positive

- Le plafond réel est désormais localisé et chiffré : qualité du front-end
  (pureté des tours, fusion de voix), pas le modèle de regroupement.
- Benchmark comparatif multi-reconstructeur (`both`) disponible pour
  mesurer objectivement toute amélioration amont.
- Métrique WER non-overlap honnête (absence de mesure ≠ 1.0).

### Negative

- Deux reconstructeurs à tester/maintenir (un seul par défaut).
- Le jalon M6 reste non validé sur son critère d'origine.

## Reconsideration criteria

Relancer l'évaluation `--reconstructor both` (critère spec inchangé) dès
qu'une itération amont satisfait simultanément : DER stable (~0,12), voix
non fusionnées inter-conversations sur tous les datasets seedés, et WER
toutes-paires < 0,7 — typiquement via séparation efficace (ADR-0008) ou
diarization multi-voix robuste à l'entrelacement.
