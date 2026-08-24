# ADR-0005: Reconstruction heuristique — scoring de paires + chaînage

## Status

Accepted

## Context

Le cœur du projet : regrouper des énoncés en conversations. Avant
d'apprendre un modèle (phase 6 : graphe, GNN), il faut une baseline simple,
interprétable et mesurable.

## Decision

Baseline heuristique :

1. Candidats : paires (i,j) avec gap ≤ max_gap (~30 s).
2. Score = w_t·exp(−Δt/τ) + w_a·alternance(locuteurs différents)
   + w_s·cosinus(embeddings sémantiques multilingues des textes).
3. Arêtes retenues si score ≥ ε et j ∈ meilleurs successeurs de i.
4. Composantes connexes (union-find) → conversations triées par temps ;
   participants = locuteurs uniques.

Poids et seuils dans la config. Les embeddings texte viennent d'une
interface `TextEmbedder` (mock déterministe en test).

## Alternatives considered

- Clustering direct sur embeddings concaténés : ignore la structure
  temporelle séquentielle.
- Modèle appris tout de suite : pas de baseline de référence, pas de données.

## Consequences

### Positive

- Simple, rapide, interprétable ; chaque signal pondérable individuellement.

### Negative

- Sensible aux interruptions rapides entre conversations proches dans le temps.

## Reconsideration criteria

Pairwise-F1 plafonné malgré réglages → passer à la représentation graphe
(phase 6) en comparant systématiquement à cette baseline.
