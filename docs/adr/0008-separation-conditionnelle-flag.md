# ADR-0008: Séparation conditionnelle SepFormer derrière flag, désactivée par défaut

## Status

Accepted — renforcé par N4 (trois itérations ASR-par-tige évaluées, toutes
dégradent WER overlap par rapport à OFF)

## Context

ADR-0004 a posé l'interface `Separator` + passthrough en attendant de
chiffrer le coût WER des zones d'overlap. Ce coût étant établi
(WER overlap 0,80 vs 0,89 non-overlap, benchmark N2), l'itération N3 a
intégré une séparation conditionnelle réelle : SepFormer whamr16k sur les
seules zones ≥2 locuteurs, tige assignée à chaque tour par similarité
cosinus entre un embedding de référence du locuteur (centroïdes ECAPA du
diarizer, sinon extrait exclusif du tour) et les embeddings des tiges,
avec double garde (similarité minimale `assign_min_sim`, marge sur la
deuxième meilleure `assign_min_margin`) ; mix conservé si la référence est
dégénérée ou qu'aucune tige ne passe les seuils.

Mesure N3 (4 datasets seedés 1234…1237, whamr16k, 2-stems) :

| Métrique | OFF | ON splice |
|---|---|---|
| WER overlap | **0.768** | 0.891 |
| WER non-overlap | **0.571** | 0.571 |
| ARI | 0.258 | 0.258 |
| pairwise_F1 | 0.458 | 0.500 |

Trois approches ASR-par-tige évaluées en N4 (splice in-place, segments
composite, tige pure) : toutes dégradent le WER overlap. La cause racine
est que SepFormer num_speakers=2 sur4 locuteurs réels produit des tiges
contenant chacune ~2 locuteurs, rendant l'assignation et la transcription
par tige non fiables.

## Decision

La séparation conditionnelle reste intégrée mais **désactivée par défaut**
(`separation.enabled: false`), activable par CLI (`--separate`). Le
remplacement audio des zones d'overlap par la tige assignée n'est pas
retenu comme comportement standard : il dégrade le WER overlap malgré le
gain de structure pour la reconstruction.

## Alternatives considered

- Activer la séparation par défaut : rejeté, WER overlap dégradé.
- Supprimer le code de séparation : rejeté, l'ARI/NMI progressent avec ON
  et l'infrastructure servira si un meilleur modèle de séparation arrive.
- ASR par tige (N4) : rejeté, trois variantes testées (splice, composite,
  pure) toutes dégradent le WER. Cause : tiges 2-speakers sur4 locuteurs.
- Séparation systématique : rejeté, coût GPU inutile hors overlap.

## Consequences

### Positive

- Le coût exact de la substitution tige-à-l'ASR est désormais connu.
- Le regroupement gagne ~0,06 ARI / 0,10 NMI quand ON est actif.
- Chemin code prêt pour un ASR par tige ou un meilleur modèle.

### Negative

- Deux chemins de pipeline à tester/maintenir (OFF par défaut).
- WER overlap ON reste supérieur à OFF même après trois itérations N4.
- Le gap WER (~0.12) est trop important pour que des ajustements de seuils
  le comblent — il faut un meilleur modèle de séparation (≥4 speakers).

## Reconsideration criteria

Activer par défaut si WER overlap(ON) < WER overlap(OFF). Leviers
pertinents : modèle de séparation à N speakers réels (pas 2), ASR
par tige avec sélection texte-level après segmentation VAD dans la tige,
ou fine-tuning Whisper sur tiges SepFormer.
