# ROADMAP — Conversation Deconvolution

Feuille de route opérationnelle dérivée du README (phases 0→8).
Les jalons M0→M5 sont le périmètre courant ; M6→M8 sont la perspective.

Chaque jalon a des **livrables** et un **critère d'acceptation** mesurable.
Principe directeur : baseline fonctionnelle → mesure → bottleneck → amélioration.

---

## M0 — Architecture & cadrage *(README Phase 0)*

**Livrables**

- [x] Repository initialisé, structure `src/`, `tests/`, `configs/`, `data/`, `docs/`.
- [x] ADR 0001→0007 (python/uv, diarisation DIY, ASR, séparation, reconstruction,
      données synthétiques, évaluation).
- [x] Formats `Utterance` / `Conversation` figés dans `core/types.py`
      (schéma JSON identique côté pipeline et vérité terrain).
- [x] Métriques définies : VAD F1 · DER · WER · pairwise-F1 / ARI / NMI.
- [x] Config YAML versionnée (`configs/default.yaml`) + seeds reproductibles.

**Acceptation :** `uv sync && make lint test` verts sur un checkout neuf.

---

## M1 — Baseline audio → transcription *(Phase 1)*

**Livrables**

- [x] Chargement audio mono 16 kHz float32.
- [x] VAD Silero avec probabilités frame-level.
- [x] ASR faster-whisper GPU par segment.
- [x] Export JSON horodaté + timeline PNG.

**Acceptation :** `deconvolute run meeting.wav -o out.json` produit une
transcription horodatée exploitable sur un fichier réel.

---

## M2 — Diarisation *(Phase 2)*

**Livrables**

- [x] Embeddings locuteur ECAPA-TDNN par segment speech.
- [x] Clustering agglomératif cosine (k auto ou `--num-speakers`).
- [x] Timeline locuteurs fusionnée + utterances attribuées (IoU).
- [x] DER calculé contre vérité terrain.

**Acceptation :** `timestamp + speaker + text` par énoncé ; DER mesuré
sur dataset synthétique.

---

## M3 — Overlap & séparation *(Phase 3)*

**Livrables**

- [x] Détection des zones ≥2 locuteurs simultanés (balayage timeline).
- [x] Interface `Separator` + baseline passthrough.
- [x] Rapport WER overlap vs non-overlap (chiffrer le problème avant
      d'intégrer un vrai modèle de séparation).

**Acceptation :** les zones de chevauchement sont identifiées et leur coût
WER est quantifié dans le rapport de benchmark.

---

## M4 — Reconstruction des conversations *(Phase 4)*

**Livrables**

- [x] Features de relation entre paires d'énoncés (gap, alternance,
      similarité sémantique multilingue).
- [x] Scoring + chaînage glouton + union-find → conversations.
- [x] Évaluation pairwise-F1 / ARI / NMI contre GT.

**Acceptation :** sur datasets synthétiques à ≥2 conversations parallèles,
la reconstruction regroupe significativement mieux que le hasard
(pairwise-F1 > baseline aléatoire documentée).

---

## M5 — Dataset synthétique & benchmark *(Phase 5)*

**Livrables**

- [x] Générateur seedé : conversations parallèles, voix Piper FR distinctes,
      SNR, gains, taux d'overlap contrôlés.
- [x] Vérité terrain complète (conversations, locuteurs, timings, textes).
- [x] Commande `deconvolute benchmark` : N datasets → pipeline → rapport
      Markdown agrégé.

**Acceptation :** `make benchmark` régénère données + métriques de façon
reproductible (même seed ⇒ mêmes chiffres).

---

## M6 — Graphe de conversations *(Phase 6)*

**Livrables**

- [x] Infrastructure complète : features de paire ordonnées, entraîneur LR
      avec persistance JSON, commande `deconvolute train` (seeds dédiés),
      `GraphReconstructor` (Louvain), sélection `--reconstructor`,
      benchmark comparatif `both`.
- [x] Itérations : retrait des features d'identité locuteur (raccourci),
      balayages seuil × résolution, agrégation voix-niveau, amélioration
      front-end (`context_pad 0`, métrique WER honnête).
- [ ] ~~Battre la baseline heuristic sur le benchmark seedé~~ — **non
      atteint** : l'information « conversation » est détruite en amont
      (contamination inter-conversations, fusion de voix) ; voir ADR-0009.

**Acceptation :** non validée — approche rejetée comme défaut, code
conservé derrière `--reconstructor graph`.

## M7 — Robustesse *(Phase 7, outlook)*

Bruit réaliste, réverbération (RIR), interruptions intra-conversation,
locuteurs proches/éloignés, longues durées, sujets changeants.

## M8 — Démonstrateur *(Phase 8, outlook)*

Upload audio, timeline interactive, visualisation des chevauchements et des
conversations, correction manuelle des associations, export.

---

## État

| Jalon | Statut |
|---|---|
| M0 | ✅ fait |
| M1 | ✅ fait |
| M2 | ✅ fait — DER 0,091 ± 0,041 |
| M3 | ✅ fait — N3 : SepFormer conditionnel (`--separate`, défaut OFF, ADR-0008) |
| M4 | ✅ fait — pairwise-F1 0,651 / ARI 0,480 |
| M5 | ✅ fait — `deconvolute benchmark` |
| M6 | ⛔ clôturé — ADR-0009, code conservé derrière `--reconstructor graph` |

**Prochaine étape :** itérations amont qualité transcription (N1 Lite-TFNet,
N2 beam search dépendant-locuteur) et Normalisation WER (N3 déjà fait).
Séparation ON restant OFF par défaut (ADR-0008 renforcé par N4 : trois
variantes ASR-par-tige échouent sur SepFormer 2-speakers vs 4 locuteurs
réels). Piste alternative : pyannote.audio pour diarisation (non disponible
dans l'environnement actuel).
réels).
