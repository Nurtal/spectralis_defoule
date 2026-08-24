# ADR-0007: Harnais d'évaluation — métriques par étage + benchmark

## Status

Accepted

## Context

Chaque évolution doit être comparable à la baseline (principe de
mesurabilité). Les librairies dédiées (dscore, pyannote.metrics) tirent des
dépendances lourdes ou imposent leurs formats.

## Decision

Implémenter nos métriques sur grilles temporelles simples :

- VAD : précision/rappel/F1 sur grille 32 ms.
- DER : grille 10 ms, collar 0,25 s, mapping optimal locuteurs (hongrois,
  scipy) ; DER = (FA + Miss + Confusion)/total.
- WER : jiwer, énoncés GT non-overlap appariés par IoU.
- Reconstruction : appariement hongrois préd↔vrai puis pairwise-F1, ARI,
  NMI (sklearn).

Commande `deconvolute benchmark` : génère N datasets seedés, exécute le
pipeline, agrège un rapport Markdown (moyenne ± écart-type).

## Alternatives considered

- pyannote.metrics : couplage à pyannote (gated).
- dscore : format RTTM uniquement, intégration rigide.
- jiwer seul pour tout : pas de DER ni métriques de clustering.

## Consequences

### Positive

- Zéro dépendance gated ; métriques testables unitairement (cas jouets).

### Negative

- Maintien de notre propre implémentation DER (risque d'erreur : mitigé
  par tests sur exemples connus).

## Reconsideration criteria

Divergence constatée avec une référence externe lors de la validation →
basculer la métrique fautive vers la librairie de référence.
