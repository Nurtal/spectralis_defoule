# Spec — Target Speaker Extraction (TSE) for Conversation Deconvolution

## 1. Problem

After ADR-0010 the architecture ceiling is reached on synthetic data:
- DER 0.091, pairwise-F1 0.651, ARI 0.480 (4 speakers, seeds 1‑4)
- SepFormer limited to 2 stems → degrades WER overlap (0.807 vs 0.768 OFF) on 4 real speakers
- Next step per roadmap: N1 "Lite‑TFNet separation" — but no public pretrained
  model with >2 stems exists. Goal: improve separation quality → lower WER overlap.

## 2. Chosen approach

**Option A — Custom lightweight TSE, conditionné sur embeddings ECAPA 192‑d.**
- Modèle STFT‑domain FiLM, ~3 M paramètres, loss SI‑SDR
- Conditionnement par embedding ECAPA (modèle gelé speechbrain/spkrec-ecapa-voxceleb)
- N speakers sans limite (limite seulement le nombre de voix Piper disponibles = 6)
- Données synthétiques Piper déterministes, bruit band‑limité SNR ∈ [10,20] dB
- Pas de dépendance externe lourde ; s'inscrit dans l'ADN DIY du projet

## 3. Architecture du modèle (src/conversation_deconvolution/tse/model.py)

| Composant | Détails |
|---|---|
| Entrée | mix mono 16 kHz float32 |
| STFT | n_fft 512, hop 256, fenêtre hann, spectrogramme real/imag emb |
| Encodeur | Blocs conv2d résiduels (3 couches, ~1.5 M params) |
| Conditionnement | FiLM (γ, β prédits depuis embedding ECAPA 192‑d, injecté dans chaque bloc) |
| Décodeur | 3 blocs conv résiduels + masque sigmoid |
| Sortie | Masque appliqué au STFT → ISTFT → waveform |
| Loss | SI‑SDR waveform‑domain (fonction torch `sdr`) |
| Optim | AdamW lr=3e‑4, grad‑clip 1.0 |
| Checkpoint | `models/tse/model.pt` + `models/tse/hparams.yaml` |

## 4. Génération de données d'entraînement (src/conversation_deconvolution/tse/dataset.py)

- **Dataset en ligne** (pas de fichier sur disque) — chaque batch échantillonné à la volée
- Chaque batch :
  1. K = rng.integers(2, min(5, N_speakers_known)+1) locuteurs parmi les 6 Piper
  2. Génération des utterances + timestamps (scénarios seedés, gaps exponentiels)
  3. **Mix** = somme sur timeline + bruit band‑limité SNR ∈ [10, 20] dB
  4. **Cible** = signal du locuteur ciblé seul (stems additionnés sur la timeline du batch)
  5. **Reference embedding** = ECAPA 192‑d moyenné sur 1‑2 utterances propres du même speaker **non utilisées dans le mix** (mimique `_exclusive_ref` à l'inférence)
- Mise en cache TTS : `models/piper/samples/` (déjà présent, deterministe md5)

## 5. Intégration pipeline (src/conversation_deconvolution/pipeline.py)

### Sélecteur de backend YAML

```yaml
separation:
  enabled: false
  backend: sepformer        # passthrough | sepformer | tse
  model_path: models/tse/model.pt   # spécifique tse
```

### Factory `build_pipeline`

```python
if config.separation.backend == "tse":
    separator = TseSeparator(config.separation)
elif config.separation.backend == "sepformer":
    separator = SepformerSeparator(config.separation)
else:
    separator = PassthroughSeparator()
```

### Extension interface `separate()`

```python
def separate(self, mix, regions, speaker_refs: dict | None = None) -> SeparationResult
```

- `speaker_refs = {label: centroid}` construit depuis `diarizer.speaker_centroids_` (fallback `_exclusive_ref` si absent)
- `TseSeparator` retourne `SeparatedRegion.stems` indexés dans l'ordre trié des clés `speaker_refs` → un stem par locuteur connu (N sans limite)
- `_assign_best_stem` garde son rôle garde‑fou : si similarité insuffisante → mix conservé
- Métadonnée `num_speakers = len(speaker_refs)` ajoutée à `SeparationResult.meta`

### Comptage locuteurs (optionnel, section 3.3)

Deux approches :
- **A) Diarization directe** : `num_speakers = len(diarizer.speaker_centroids_)` — zéro coût additionnel, exposé dans le rapport benchmark
- **B) Estimateur BIC sur mix** : embeddings frame‑level ECAPA + K‑means K∈{2…6}, sélection BIC — intégré optionnel dans `separate()` → `result.meta["num_speakers_estimated"]`

## 6. Commandes CLI (src/conversation_deconvolution/cli.py)

```bash
# Entraînement TSE (8 datasets, 30 epochs ~2‑3h GPU)
uv run deconvolute train-tse --datasets 8 --epochs 30 --out models/tse/model.pt

# Inférence avec backend TSE
deconvolute run meeting.wav -o out.json --separate --separation-backend tse

# Benchmark comparatif OFF vs ON‑TSE
deconvolute benchmark --datasets 3 --separate --separation-backend tse -o reports/benchmark_tse.md
```

## 7. Critères d'acceptation (ADR‑0011)

| Critère | Seuil |
|---|---|
| SI‑SDR validation | ≥ 5 dB moyen |
| WER overlap (benchmark, 4 speakers) | < 0.807 (baseline SepFormer OFF) ; cible ≈ 0.75 |
| Taille modèle | ≤ 4 M paramètres |
| Inférence CPU 16 kHz mono | ≤ 200 ms |
| Rapport benchmark | inclut `"Nb speakers"` (entier) |

## 8. Tests (résumé)

| Type | Fichier | Couverture |
|---|---|---|
| Unitaire | `tests/unit/test_tse_dataset.py` | Génération batch infinie, shapes, pas NaN |
| Unitaire | `tests/unit/test_tse_model.py` | Forward, shapes SI‑SDR, comptage params |
| Unitaire | `tests/unit/test_tse_separator.py` | `TseSeparator` avec/without `speaker_refs`, fallback mix |
| Intégration | `tests/integration/test_e2e_synthetic.py` | +marqueur `@pytest.mark.tse_slow` (GPU) |
| Config | `tests/unit/test_config.py` | Section `separation.backend` + `model_path` |
| Pipeline | `tests/unit/test_pipeline_separation.py` | Extension test existant : `speaker_refs` simulé |

## 9. Fichiers touchés

| Fichier | Changement |
|---|---|
| `core/config.py` | +`separation.backend`, +`separation.model_path`, section `tse:` hyperparams |
| `configs/default.yaml` | idem |
| `cli.py` | option `--separation-backend`, commande `train-tse` |
| `pipeline.py` | factory backend, construction `speaker_refs`, comptage `num_speakers` |
| `tse/__init__.py` | exports `TseModel, TseDataset, TseSeparator` |
| `tse/model.py` | architecture légère FiLM + Si‑SDR |
| `tse/dataset.py` | dataset en ligne, augmentation TTS / bruit |
| `tse/train.py` | boucle d'entraînement, checkpoint, early‑stopping |
| `separation/tse_separator.py` | `separate(mix, regions, speaker_refs=None)` + metadata num_speakers |
| `evaluation/wer.py` | clé `"Nb speakers"` dans dict de rapport |
| `Makefile` | cible `train-tse` |
| `tests/` | nouveaux fichiers unitaires + intégration slow |

---

*Fin de spec. Prochaine étape : révision par l'utilisateur, puis invocation de la skill `writing-plans` pour créer le plan d'implémentation détaillé.*