# ADR-0008: Séparation conditionnelle SepFormer derrière flag, désactivée par défaut

## Status

Accepted (supersedes la partie « report » de ADR-0004, dont l'option
d'intégration est désormais levée et mesurée)

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

Mesure (4 datasets seedés 1234…1237) :

| Métrique | OFF | ON |
|---|---|---|
| WER overlap | **0.8023** | 1.0021 |
| ARI | 0.2011 | **0.2577** |
| NMI | 0.4367 | **0.5361** |

## Decision

La séparation conditionnelle reste intégrée mais **désactivée par défaut**
(`separation.enabled: false`), activable par CLI (`--separate`). Le
remplacement audio des zones d'overlap par la tige assignée n'est pas
retenu comme comportement standard : il dégrade le WER overlap malgré le
gain de structure pour la reconstruction.

## Alternatives considered

- Activer la séparation par défaut : rejeté, le WER overlap se dégrade.
- Supprimer le code de séparation : rejeté, l'ARI/NMI progressent avec ON
  et l'infrastructure servira aux itérations suivantes.
- Séparation systématique sur tout l'audio : rejeté d'emblée, coût GPU
  inutile hors overlap et hors hypothèse du design conditionnel.

## Consequences

### Positive

- Le coût exact de la substitution tige-à-l'ASR est désormais connu.
- Le regroupement gagne ~0,06 ARI / 0,10 NMI quand ON est actif.
- Chemin code prêt pour un ASR par tige ou un meilleur modèle.

### Negative

- Deux chemins de pipeline à tester/maintenir (OFF par défaut).
- Résultat ON négatif sur le WER overlap en l'état.

## Reconsideration criteria

Activer par défaut si une itération rend le WER overlap(ON) < WER
overlap(OFF), typiquement via ASR par tige avec sélection texte-level,
modèle de séparation adapté à la parole propre, ou fine-tuning ; suivre
alors SI-SDR des tiges pour distinguer qualité séparation vs attribution.
