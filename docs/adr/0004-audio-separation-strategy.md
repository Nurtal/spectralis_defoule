# ADR-0004: Séparation conditionnelle reportée, baseline passthrough

## Status

Accepted

## Context

La séparation de sources n'a de sens qu'une fois le problème quantifié :
sans mesure WER overlap vs non-overlap, impossible de savoir ce qu'elle
apporte (README : ne pas sur-ingénieriser trop tôt).

## Decision

Phase courante : interface `Separator` avec implémentation passthrough
(le mix est rendu tel quel). Le pipeline marque les zones d'overlap issues
de la timeline diarisation. Le rapport de benchmark chiffre le coût WER des
zones overlap ; l'intégration d'un vrai modèle (ex. SepFormer/Demucs) se
fera en phase 3 pleine, en comparant séparation systématique vs
conditionnelle.

## Alternatives considered

- Intégrer immédiatement un modèle de séparation : coût fort, bénéfice non mesuré.
- Ne pas définir d'interface : rendrait le remplacement invasif plus tard.

## Consequences

### Positive

- Pipeline complet dès maintenant ; décision future guidée par la mesure.

### Negative

- WER dégradé sur les zones d'overlap en attendant.

## Reconsideration criteria

Si le WER overlap s'avère négligeable sur nos données, la priorité de la
phase 3 pleine sera revue à la baisse.
