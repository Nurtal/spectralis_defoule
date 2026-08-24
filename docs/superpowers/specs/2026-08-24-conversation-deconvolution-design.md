# Conversation Deconvolution — Design (Phase 0→5)

**Date :** 2026-08-24
**Statut :** Approuvé (design présenté et validé en session)
**Périmètre :** Phases 0 à 5 du README — architecture, pipeline end-to-end
fonctionnel, dataset synthétique, évaluation. Phases 6→8 hors périmètre.

## 1. Objectif

Assembler des composants existants derrière des interfaces modulaires pour
produire une **baseline mesurable** du pipeline :

```
audio → VAD → diarisation → (séparation: passthrough) → ASR → utterances
      → reconstruction de conversations → JSON + timeline
```

Principes respectés : modularité, mesurabilité, reproductibilité,
local-first, ADR-driven, pas de sur-ingénierie.

## 2. Stack

| Rôle | Choix | Alternative écartée |
|---|---|---|
| Langage / build | Python 3.12 + uv | poetry |
| VAD | Silero VAD v6 (`silero-vad`) | pyannote VAD (gated) |
| Embeddings locuteur | SpeechBrain ECAPA-TDNN (VoxCeleb) | pyannote embeddings (gated) |
| Clustering | sklearn AgglomerativeClustering (cosine) | spectral (plus tard si besoin) |
| ASR | faster-whisper (CTranslate2, GPU float16), défaut `small` | whisper openai (lent), whisperX (tire pyannote) |
| Séparation | Passthrough (identité) — interface prête pour phase 3 | SepFormer etc. (reporté) |
| Embeddings sémantiques | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` | LLM API (pas local-first) |
| TTS synthétique | piper-tts 1.7, voix fr_FR (siwis, tom, upmc, …) | edge-tts (cloud), XTTS (gated) |
| Éval | scipy (hongrois), jiwer (WER), sklearn (ARI/NMI) | dscore, sb_wer |
| CLI | typer + rich | argparse |

GPU : 2× TITAN RTX (sm_75) supportées par torch cu124 et CTranslate2.
Tous les modèles sont téléchargeables sans compte ni token HF.

## 3. Architecture

### 3.1 Interfaces (protocoles Python, un par étage)

```python
class VadModel:        detect(audio_16k) -> VadResult(segments, frame_probs)
class SpeakerEmbedder: embed(segment_audio) -> np.ndarray  # L2-normalisé
class Clusterer:       fit_predict(embeddings, n_speakers=None) -> labels
class AsrEngine:       transcribe(segment_audio, language=None) -> AsrResult(text, confidence, language)
class Separator:       separate(mix, regions) -> list[SourceAudio]   # passthrough en baseline
class TextEmbedder:    encode(texts) -> np.ndarray
class Reconstructor:   reconstruct(utterances) -> list[Conversation]
class TtsEngine:       synthesize(text, voice) -> AudioSegment
```

Chaque étage est remplaçable indépendamment (exigence README).

### 3.2 Layout

```
src/conversation_deconvolution/
├── core/          types.py (Segment, Utterance, Conversation, VadResult…),
│                  config.py (PipelineConfig + sous-configs, YAML)
├── audio/         loader.py (mono 16 kHz float32), vad.py (Silero)
├── diarization/   embeddings.py (ECAPA), clusterer.py (agglomératif), diarizer.py
├── separation/    passthrough.py
├── asr/           faster_whisper_asr.py
├── conversation/  features.py, reconstructor.py, export.py, viz.py (timeline)
├── evaluation/    vad_metrics.py, der.py, wer.py, clustering_metrics.py, report.py
├── synthetic/     tts.py (Piper), scenario.py, mixer.py, generator.py
├── pipeline.py    orchestrateur
└── cli.py         run / synth / evaluate / viz / benchmark

tests/unit/  tests/integration/
configs/default.yaml
data/{raw,processed,synthetic}/   (gitignorés)
docs/adr/  docs/superpowers/{specs,plans}/
```

## 4. Modèle de données

`Utterance(id, speaker, start, end, text, confidence, language)` ;
`Conversation(id, participants, utterances)` ; schéma JSON d'export identique
à l'exemple du README. La vérité terrain synthétique utilise le **même
schéma**, permettant comparaison directe préd↔GT.

## 5. Pipeline (détails algorithmiques)

1. **Load** : soundfile → mono, 16 kHz, float32 [-1,1].
2. **VAD Silero** : segments speech + probabilités par frame (hop 512).
3. **Embeddings ECAPA** : par segment speech (durée min 0.4 s), 192-d, L2.
4. **Clustering agglomératif** : linkage average, distance cosinus ;
   `n_speakers` si fourni sinon seuil de distance (~0.75, configurable).
5. **Timeline locuteurs** : fusion des segments consécutifs même label ;
   détection overlap par balayage (≥2 locuteurs simultanés).
6. **ASR faster-whisper** : par segment diarisé (+0.25 s de contexte),
   détection langue auto, confiance = moyenne exp(logprob).
7. **Fusion** : assignation ASR↔diarisation par IoU maximal → Utterances.
8. **Reconstruction** : pour paires (i,j) avec gap ≤ max_gap :
   score = w_t·exp(−Δt/τ) + w_a·alternance + w_s·cos(emb_texte_i, emb_texte_j)
   Arête retenue si score ≥ ε et j ∈ meilleurs candidats de i ; composantes
   connexes (union-find) → conversations triées par temps.
9. **Sorties** : JSON conversations + PNG timeline.

## 6. Dataset synthétique

Générateur seedé : n_conversations, locuteurs/conv, durée cible, gaps de
tours (loi exponentielle), longueurs d'énoncés (mots), SNR, gains relatifs.
Voix Piper FR distinctes assignées par locuteur ; chaque ligne de dialogue
synthétisée puis placée sur la timeline ; mixage numpy + bruit blanc filtré
au SNR exact ; sorties `mixed.wav` (16 k) + `ground_truth.json`.

## 7. Évaluation

- **VAD** : P/R/F1 sur grille 32 ms vs GT.
- **DER** : grille 10 ms, collar 0,25 s, mapping optimal hongrois,
  DER = (FA + Miss + Confusion)/durée totale.
- **WER** : jiwer, GT non-overlap appariées aux prédictions par IoU.
- **Reconstruction** : appariement hongrois préd↔vrai, puis pairwise-F1,
  ARI, NMI.
- **Benchmark** : N datasets générés → exécution pipeline → rapport Markdown
  agrégé (métriques moyennes ± écart-type).

## 8. Tests & qualité

- Unitaires rapides sans modèles : maths (SNR/mixage, DER, WER, features,
  union-find, balayage overlap, IoU), scénario seedé déterministe, schéma
  d'export round-trip. Interfaces mockées (fakes déterministes).
- Intégration marquée `slow` : génération mini-dataset réel (Piper) +
  pipeline complet (whisper tiny) sur ~20 s d'audio ; assertions schéma +
  nb locuteurs ≥ 2.
- ruff (lint + format), Makefile (`make lint test test-slow benchmark`).

## 9. Risques

| Risque | Mitigation |
|---|---|
| API piper 1.7 récente mal documentée | Interface TtsEngine isolée ; fallback binaire CLI |
| Téléchargements lourds (~2–3 Go) | Cache HF local ; tests unitaires sans modèles |
| Reconstruction heuristique faible si ASR bruité | Acceptable en baseline : mesuré par le benchmark, amélioration phase 4+ |
| Qualité clustering dépendante du seuil | Option `--num-speakers`, seuil dans config |

## 10. Décisions couvertes par ADR

0001 python/uv · 0002 stack diarisation DIY · 0003 ASR faster-whisper ·
0004 séparation conditionnelle reportée · 0005 reconstruction heuristique ·
0006 données synthétiques Piper FR · 0007 harnais d'évaluation.
