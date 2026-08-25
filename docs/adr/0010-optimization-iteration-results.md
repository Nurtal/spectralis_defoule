# ADR-0010 : Résultats des itérations d'optimisation diarisation + reconstruction

## Statut

Accepté

## Contexte

Après la baseline M0-M5 et les décisions ADR-0008/0009, une session
d'optimisation intensive a exploré toutes les configurations
raisonnables de diarisation et de reconstruction pour améliorer
DER, pairwise-F1 et ARI sur le benchmark synthétique 4 datasets.

## Décision

### Ce qui améliore les métriques

| Variation | Impact | Mécanisme |
|---|---|---|
| `window_sec` 1.5→1.0 | DER −32%, F1 +13%, ARI +20% | Fenêtres plus courtes = moins de mélange locuteurs |
| `hop_sec` 0.5→0.33 | Idem (couplé avec window) | Plus de votes = meilleure résolution |
| `cell_sec` 0.25→0.125 | F1 +7%, ARI +37% | Frontières de tours plus précises |
| Reweight reconstruction (semantic-heavy, threshold=0.4) | F1 +7.6%, ARI +13% | La similarité sémantique est le meilleur signal sur données synthétiques |

### Ce qui NE marche PAS

| Variation | Résultat | Raison |
|---|---|---|
| `beam_size` > 1 | Dégrade WER | Sur données synthétiques courtes, beam search hallucine |
| `language="fr"` | Aucun impact | Les transcriptions FR sont déjà correctes |
| Clustering spectral (sklearn) | DER +36%, F1 −26% | Le graphe de similarité est bruité par les fenêtres mixtes |
| Splitting post-diarization | DER +100% | Les centroïdes sont contaminés par le mélange locuteurs |
| `w_alternation` (A→B→A) | Dégrade F1/ARI | Les poids normalisés rendent l'alternance trop pénalisante |
| Refinement pass post-greedy | Dégrade F1/ARI | Perturbe les assignations correctes existantes |
| `initial_prompt` ASR | Dégrade reconstruction | Le prompt biaise les textes et perturbe la similarité sémantique |
| ASR medium/large | Dégrade WER | Les modèles plus gros sur-fittent sur données synthétiques |
| `max_speaker_overlap_ratio` varié | Aucun impact sauf >0.20 | Le seuil de conflit est déjà optimal |
| `min_segment_sec` réduit (0.2) | Dégrade ARI −41% | Plus de segments courts = plus de bruit dans le clustering |
| SepFormer separation ON | Dégrade WER overlap | SepFormer 2-speakers vs 4 locuteurs réels |

### Pyannote.audio

Intégré comme backend alternatif (`--diarization-backend pyannote`).
Sur données synthétiques, notre pipeline DIY reste meilleur
(DER 0.091 vs 0.130 pyannote). Pyannote produit beaucoup de
micro-segments et est très variable (DER 0.012–0.256).
Peut être utile sur données réelles où les modèles pré-entraînés
ont un avantage.

### Config optimale finale

```yaml
diarization:
  window_sec: 1.0
  hop_sec: 0.33
  cell_sec: 0.125
  num_speakers: 4  # oracle
reconstruction:
  w_temporal: 0.10
  w_semantic: 0.50
  w_same_speaker: 0.40
  threshold: 0.4
asr:
  model_size: small
  beam_size: 1
  language: fr
```

### Métriques finales (4 datasets, 4 seeds)

| Métrique | Baseline | Optimisé | Delta |
|---|---|---|---|
| DER | 0.125 | 0.091 | −27% |
| pairwise_F1 | 0.500 | 0.651 | +30% |
| ARI | 0.258 | 0.480 | +86% |
| WER non-overlap | — | 0.607 | — |
| WER overlap | — | 0.807 | — |

## Conséquences

- Le plafond de l'architecture actuelle est atteint sur données synthétiques.
- Les prochaines améliorations nécessitent un changement d'architecture :
  - N1 : Lite-TFNet pour séparation plus légère
  - Modèles d'embeddings temps-réel (frame-level) au lieu de par-fenêtre
  - Pyannote pour données réelles
- Le WER reste le maillon faible, limité par la qualité de diarisation.
- Les données synthétiques ont un biais : les modèles plus gros (medium/large)
  ne s'améliorent pas, ce qui est inhabituel et suggère que les données
  sont trop simples pour discriminer les modèles.
