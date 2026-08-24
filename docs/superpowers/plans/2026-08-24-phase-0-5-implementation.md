# Conversation Deconvolution Phase 0–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline end-to-end mesurable : audio multi-conversations → VAD → diarisation DIY → ASR → reconstruction heuristique des conversations → JSON + évaluation sur datasets synthétiques Piper FR.

**Architecture:** Modules interchangeables derrière des protocoles (`VadModel`, `SpeakerEmbedder`, `Clusterer`, `AsrEngine`, `Separator`, `TextEmbedder`, `Reconstructor`, `TtsEngine`), orchestrateur injectable (testable avec des fakes), vérité terrain au même schéma JSON que l'export.

**Tech Stack:** Python 3.12, uv, torch cu124, silero-vad 6, speechbrain 1.1, faster-whisper 1.2, sentence-transformers, piper-tts 1.7, numpy/scipy/sklearn/soundfile, jiwer, typer/rich, matplotlib, pytest, ruff.

**Spec:** docs/superpowers/specs/2026-08-24-conversation-deconvolution-design.md

## Global Constraints

- Python 3.12, gestion uv (`uv sync` doit suffire sur checkout neuf).
- Torch depuis l'index `https://download.pytorch.org/whl/cu124`.
- Audio canonique interne : mono, 16000 Hz, float32, plage [-1, 1].
- Aucune dépendance gated HF ; aucun appel cloud.
- Pas de commentaires dans le code (style repo) ; docstrings une ligne autorisées.
- Tests unitaires < 10 s sans téléchargement ; intégration sous marqueur `slow`.
- Chaque tâche se termine par un commit vert (tests + ruff).

---

### Task 1: Scaffolding projet

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `Makefile`, `configs/default.yaml`, `src/conversation_deconvolution/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `data/{raw,processed,synthetic}/.gitkeep`

**Interfaces:**
- Produces: paquet installable `conversation_deconvolution` (src layout), cibles `make lint`, `make test`.

- [ ] **Step 1:** `uv init --lib --name conversation-deconvolution --python 3.12 .` puis écrire `pyproject.toml` :

```toml
[project]
name = "conversation-deconvolution"
version = "0.1.0"
description = "Déconvolution de conversations multi-locuteurs à partir d'un enregistrement audio unique"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0", "scipy>=1.13", "scikit-learn>=1.5",
  "soundfile>=0.12", "torch>=2.4", "torchaudio>=2.4",
  "silero-vad>=5.1", "speechbrain>=1.1", "faster-whisper>=1.2",
  "sentence-transformers>=3.0", "piper-tts>=1.7",
  "jiwer>=3.0", "pyyaml>=6.0", "typer>=0.12", "rich>=13.0", "matplotlib>=3.9",
]
[project.scripts]
deconvolute = "conversation_deconvolution.cli:app"
[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6"]
[build-system]
requires = ["uv_build>=0.7"]
build-backend = "uv_build"
[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]
torchaudio = [{ index = "pytorch-cu124" }]
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
[tool.ruff]
line-length = 95
src = ["src", "tests"]
[tool.pytest.ini_options]
testpaths = ["tests/unit"]
markers = ["slow: tests lourds nécessitant modèles et réseau"]
```

- [ ] **Step 2:** `.gitignore` (`.venv/`, `data/**` sauf `.gitkeep`, `__pycache__/`, `*.egg-info/`, `models/`, `reports/`), Makefile :

```makefile
lint:
	uv run ruff check src tests && uv run ruff format --check src tests
format:
	uv run ruff format src tests && uv run ruff check --fix src tests
test:
	uv run pytest -q
test-slow:
	uv run pytest -q -m slow tests/integration
benchmark:
	uv run deconvolute benchmark --datasets 3 --out reports/benchmark.md
.PHONY: lint format test test-slow benchmark
```

- [ ] **Step 3:** `configs/default.yaml` (valeurs = défauts des dataclasses des Tasks 3+) ; `uv sync` ; vérifier `uv run python -c "import torch; print(torch.cuda.is_available())"` → True.

- [ ] **Step 4:** Commit : `chore: scaffold uv project, configs, make targets`

### Task 2: Types cœur + schéma JSON

**Files:**
- Create: `src/conversation_deconvolution/core/types.py`
- Test: `tests/unit/test_types.py`

**Interfaces:**
- Produces :
  - `Segment(start: float, end: float)` frozen, propriété `duration`.
  - `VadResult(segments: list[Segment], frame_probs: np.ndarray, frame_rate: float)`.
  - `SpeakerTurn(speaker: str, start: float, end: float)`.
  - `Utterance(id: str, speaker: str|None, start: float, end: float, text: str = "", confidence: float|None = None, language: str|None = None)`.
  - `Conversation(id: str, participants: list[str], utterances: list[Utterance])`.
  - `TranscriptResult(utterances: list[Utterance], conversations: list[Conversation], overlaps: list[Segment])`.
  - `utterance_to_dict / utterance_from_dict`, `conversation_to_dict / conversation_from_dict`, `result_to_dict / result_from_dict`, `conversations_to_result(list[Conversation]) -> TranscriptResult`.

- [ ] **Step 1:** Test échouant :

```python
def test_utterance_round_trip():
    u = Utterance("u1", "speaker_01", 1.0, 2.5, "salut", 0.9, "fr")
    assert utterance_from_dict(utterance_to_dict(u)) == u

def test_conversation_schema_matches_readme():
    c = Conversation("conversation_01", ["speaker_01"],
                     [Utterance("u1", "speaker_01", 12.4, 15.2, "Tu viens demain ?")])
    d = conversation_to_dict(c)
    assert set(d) == {"id", "participants", "utterances"}
    assert set(d["utterances"][0]) >= {"speaker", "start", "end", "text"}

def test_segment_duration():
    assert Segment(1.0, 2.5).duration == 1.5
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/test_types.py -q` → FAIL (module absent).
- [ ] **Step 3:** Implémenter types + conversions (dataclasses, floats natifs dans les dicts).
- [ ] **Step 4:** Run → PASS. Commit : `feat(core): data model Utterance/Conversation + JSON schema`

### Task 3: Configuration YAML

**Files:**
- Create: `src/conversation_deconvolution/core/config.py`
- Modify: `configs/default.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces : `VadConfig(threshold=0.5, min_speech_ms=250, min_silence_ms=100)` ; `DiarizationConfig(distance_threshold=0.75, min_segment_sec=0.4, num_speakers=None)` ; `AsrConfig(model_size="small", device="cuda", compute_type="float16", language=None, context_pad_sec=0.25)` ; `ReconstructionConfig(max_gap=30.0, tau=4.0, w_temporal=0.5, w_alternation=0.15, w_semantic=0.35, threshold=0.45, max_successors=2)` ; `SyntheticConfig(sample_rate=16000, snr_db=15.0, mean_gap_sec=0.8, min_words=3, max_words=14)` ; `PipelineConfig(vad, diarization, asr, reconstruction, synthetic, text_embedding_model="paraphrase-multilingual-MiniLM-L12-v2")` ; `PipelineConfig.from_yaml(path)`, `PipelineConfig.default()`.

- [ ] **Step 1:** Test :

```python
def test_default_loads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("vad:\n  threshold: 0.6\ndiarization:\n  num_speakers: 4\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.vad.threshold == 0.6
    assert cfg.diarization.num_speakers == 4
    assert cfg.asr.model_size == "small"

def test_defaults():
    assert PipelineConfig.default().reconstruction.max_gap == 30.0
```

- [ ] **Step 2:** FAIL → **Step 3:** dataclasses + chargement YAML récursif (sections partielles tolérées) → **Step 4:** PASS. Commit : `feat(core): typed YAML configuration`

### Task 4: Chargement audio

**Files:**
- Create: `src/conversation_deconvolution/audio/loader.py`, `audio/__init__.py`
- Test: `tests/unit/test_loader.py`

**Interfaces:**
- Produces : `load_audio(path: str|Path, target_sr: int = 16000) -> np.ndarray` (mono float32).

- [ ] **Step 1:** Test : écrire un WAV stéréo 44100 Hz (sinus via soundfile) ; `load_audio` → dtype float32, ndim 1, longueur ≈ durée×16000 ±1 %.

```python
def test_load_resamples_to_mono_16k(tmp_path):
    sr = 44100; t = np.linspace(0, 1.0, sr, endpoint=False)
    data = np.stack([0.5*np.sin(2*np.pi*440*t)]*2, axis=1).astype(np.float32)
    p = tmp_path/"a.wav"; sf.write(p, data, sr)
    y = load_audio(p)
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - 16000) < 160
```

- [ ] **Step 2:** FAIL → **Step 3:** soundfile read (always_2d, dtype float32) + moyenne stéréo + `scipy.signal.resample_poly` si sr≠cible → **Step 4:** PASS. Commit : `feat(audio): canonical mono 16k loader`

### Task 5: Features relationnelles + Union-Find

**Files:**
- Create: `src/conversation_deconvolution/conversation/features.py`, `conversation/__init__.py`
- Test: `tests/unit/test_features.py`

**Interfaces:**
- Consumes: `Utterance`.
- Produces :
  - `gap(a: Utterance, b: Utterance) -> float` (b.start − a.end, clampé ≥0).
  - `alternation(a, b) -> float` (1.0 si speakers différents et non-None sinon 0.0).
  - `temporal_score(a, b, tau: float) -> float = exp(-gap/tau)`.
  - `candidate_pairs(utterances, max_gap) -> list[tuple[int,int]]` (i<j triés par temps, gap ≤ max_gap).
  - `UnionFind` avec `union(i,j)`, `find(i)`, `groups() -> dict[int, list[int]]`.

- [ ] **Step 1:** Tests : gap négatif clampe à 0 ; temporal décroît avec le gap ; candidates exclut paires trop lointaines ; UF fusionne et groupe (cas 0-1-2 reliés, 3 isolé).

```python
def test_candidate_pairs_window():
    us = [U(i, s, t, t+1.0) for i,(s,t) in enumerate([("A",0),("B",1.5),("A",40)])]
    assert candidate_pairs(us, max_gap=30.0) == [(0,1),(0,2),(1,2)]
```

(gap entre fin de u1 (2.5) et début de u2 (40) = 37.5 > 30 → la paire (1,2) doit être exclue ; ajuster l'attendu à `[(0,1)]` si fenêtre stricte — décision : fenêtre stricte, attendu `[(0,1)]`.)

- [ ] **Step 2:** FAIL → **Step 3:** implémentation → **Step 4:** PASS. Commit : `feat(conversation): relational features + union-find`

### Task 6: Reconstructeur heuristique

**Files:**
- Create: `src/conversation_deconvolution/conversation/reconstructor.py`
- Test: `tests/unit/test_reconstructor.py` (+ fake `TextEmbedder` dans `tests/unit/conftest.py`)

**Interfaces:**
- Consumes: Task 5, `ReconstructionConfig`, `Conversation`.
- Produces :
  - `TextEmbedder` (protocole) : `encode(texts: list[str]) -> np.ndarray (n,d)` L2-normalisé.
  - `HashTextEmbedder(dim=64)` : déterministe (hash md5 → vecteur), pour tests.
  - `HeuristicReconstructor(embedder: TextEmbedder, config: ReconstructionConfig)` : `reconstruct(utterances: list[Utterance]) -> list[Conversation]`.
  - Algorithme : scores `w_t·temporal + w_a·alternation + w_s·cos(text)` ; arête i→j si score ≥ threshold et j parmi `max_successors` meilleurs successeurs de i ; union-find ; ids `conversation_%02d` ordonnés par première apparition ; participants par ordre d'apparition.

- [ ] **Step 1:** Tests : entrée vide → [] ; une seule conversation → 1 groupe ; deux dialogues entrelacés (A/B vocabulaire « demain café » vs C/D « réunion rapport », gaps courts intra, ~20 s inter) reconstruits en 2 groupes avec HashTextEmbedder ; ids séquentiels `conversation_01`…

- [ ] **Step 2:** FAIL → **Step 3:** implémentation (cosinus via produit scalaire sur vecteurs L2) → **Step 4:** PASS. Commit : `feat(conversation): heuristic conversation reconstructor`

### Task 7: Métriques VAD

**Files:**
- Create: `src/conversation_deconvolution/evaluation/vad_metrics.py`, `evaluation/__init__.py`
- Test: `tests/unit/test_vad_metrics.py`

**Interfaces:**
- Produces : `frame_flags(probs: np.ndarray, threshold: float) -> np.ndarray(bool)` ; `vad_prf(pred_probs, frame_rate, gt_segments: list[Segment], threshold=0.5) -> dict[str, float]` (precision/recall/f1, grille alignée probs).

- [ ] **Step 1:** Tests : probs parfaites alignées sur GT → f1=1 ; probs toutes basses → recall 0 ; moitié couverte → valeurs attendues calculées à la main.

- [ ] **Step 2:** FAIL → **Step 3:** GT flags depuis segments (bornes × frame_rate, clip) puis P/R/F1 micro → **Step 4:** PASS. Commit : `feat(evaluation): frame-level VAD metrics`

### Task 8: DER

**Files:**
- Create: `src/conversation_deconvolution/evaluation/der.py`
- Test: `tests/unit/test_der.py`

**Interfaces:**
- Consumes: `SpeakerTurn`.
- Produces : `DerResult(total: float, correct: float, miss: float, false_alarm: float, confusion: float)` avec propriété `der` ; `diarization_error_rate(ref: list[SpeakerTurn], hyp: list[SpeakerTurn], collar: float = 0.25, step: float = 0.01) -> DerResult`.

- Algorithme : grille uniforme sur [t0−collar, tmax] ; matrices de comptage ref×hyp ; mapping optimal `scipy.optimize.linear_sum_assignment(-C)` ; labels hyp translatés par φ ; comparaison frame à frame (ref −1 = silence, hyp hors mapping = FA potentielle) : correct/miss/fa/confusion cumulés ; total = durée speech ref après collar.

- [ ] **Step 1:** Tests : identiques → der≈0 ; locuteurs permutés → der≈0 (mapping) ; ref = A sur [0,10], hyp = B sur [0,10] → confusion=10, der≈1 ; hyp parlant là où ref silencieux → fa pur.

```python
def test_swapped_labels_zero_der():
    ref = [SpeakerTurn("A",0,5), SpeakerTurn("B",5,10)]
    hyp = [SpeakerTurn("X",0,5), SpeakerTurn("Y",5,10)]
    assert diarization_error_rate(ref, hyp).der == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2:** FAIL → **Step 3:** implémentation grille → **Step 4:** PASS (chaque cas ≤ 0,1 s de calcul). Commit : `feat(evaluation): grid DER with hungarian speaker mapping`

### Task 9: WER + métriques de clustering

**Files:**
- Create: `src/conversation_deconvolution/evaluation/wer.py`, `evaluation/clustering_metrics.py`
- Test: `tests/unit/test_wer.py`, `tests/unit/test_clustering_metrics.py`

**Interfaces:**
- Consumes: `Utterance`, `Conversation`.
- Produces :
  - `iou(a: Utterance|Segment, b) -> float` ; `match_by_iou(gt, pred, min_iou=0.3) -> list[tuple[Utterance, Utterance]]` (hongrois sur −IoU).
  - `wer_report(pairs: list[tuple[str,str]]) -> dict` (wer, substitutions, deletions, insertions via jiwer.process_words agrégé).
  - `labels_from_conversations(convs, keys) -> np.ndarray` ; `conversation_metrics(true_convs, pred_convs, matched_keys) -> dict` avec pairwise_f1 (formule contingence : P=R=F1 sur paires), ari, nmi (sklearn).

- [ ] **Step 1:** Tests : WER connu (« the cat sat » vs « the cat sat ») = 0 ; (« the cat » vs « the dog ») = 0.5 ; IoU chevauchement partiel ≈ 0.33 pour [0,2]/[1,2] ; partitions identiques → f1=ari=nmi=1 ; cas croisé connu → valeurs calculées à la main (2 groupes ↔ 2 groupes croisés : f1 attendue via formule paires).

- [ ] **Step 2:** FAIL → **Step 3:** implémentations → **Step 4:** PASS. Commit : `feat(evaluation): WER + conversation clustering metrics`

### Task 10: Diarisation (embeddings, clusterer, timeline)

**Files:**
- Create: `src/conversation_deconvolution/diarization/embeddings.py`, `clusterer.py`, `timeline.py`, `diarizer.py`
- Test: `tests/unit/test_timeline.py`, `tests/unit/test_clusterer.py`

**Interfaces:**
- Consumes: `VadModel`, configs Task 3.
- Produces :
  - `SileroVad(VadModel)` : `detect(audio: np.ndarray) -> VadResult` (segments via `get_speech_timestamps`, frame_probs par fenêtres 512).
  - `EcapaEmbedder(SpeakerEmbedder)` : `embed(segment: np.ndarray) -> np.ndarray(192,)` L2 (speechbrain `EncoderClassifier`, device cuda, savedir `models/speechbrain`).
  - `AgglomerativeClusterer(Clusterer)` : `fit_predict(X, n_speakers=None) -> np.ndarray(int)` (cosine average, threshold si k inconnu).
  - `merge_turns(turns, gap=0.2) -> list[SpeakerTurn]` (fusionne consécutifs même speaker) ; `overlap_regions(turns) -> list[Segment]` (balayage : zones ≥2 locuteurs actifs).
  - `SpeakerDiarizer(vad, embedder, clusterer, config).diarize(audio) -> tuple[list[SpeakerTurn], list[np.ndarray]]` (turns fusionnés + embeddings alignés aux segments bruts).

- [ ] **Step 1:** Tests purs (sans modèles) : merge_turns fusionne/fusionne-pas correctement ; overlap_regions sur timeline jouet ([A:0-2, B:1-3] → overlap [1,2] ; disjoints → []) ; AgglomerativeClusterer sépare 2 blobs gaussiens bien distincts sans k, respecte k=3 sur 3 blobs.

```python
def test_overlap_basic():
    turns = [SpeakerTurn("A",0,2), SpeakerTurn("B",1,3)]
    assert overlap_regions(turns) == [Segment(1.0, 2.0)]
```

- [ ] **Step 2:** FAIL → **Step 3:** implémentation (Silero/ECAPA wrappers paresseux : modèle chargé au premier appel) → **Step 4:** PASS. Commit : `feat(diarization): silero vad, ecapa embeddings, clustering, timeline`

### Task 11: ASR + séparation passthrough

**Files:**
- Create: `src/conversation_deconvolution/asr/faster_whisper_asr.py`, `separation/passthrough.py`
- Test: `tests/unit/test_passthrough.py`

**Interfaces:**
- Produces :
  - `AsrResult(text: str, confidence: float, language: str)` ; `FasterWhisperAsr(AsrEngine)` : `transcribe(segment: np.ndarray, language: str|None) -> AsrResult` (confiance = moyenne exp(avg_logprob)).
  - `Separator` (protocole) : `separate(mix: np.ndarray, regions: list[Segment]) -> list[np.ndarray]` ; `PassthroughSeparator` retourne [mix.copy()].

- [ ] **Step 1:** Test passthrough : identité bit-à-bit, indépendante du mix. Whisper testé en intégration uniquement.
- [ ] **Step 2:** FAIL → **Step 3:** implémentations fines → **Step 4:** PASS. Commit : `feat(asr,separation): faster-whisper engine + passthrough separator`

### Task 12: Dataset synthétique

**Files:**
- Create: `src/conversation_deconvolution/synthetic/tts.py`, `scenario.py`, `mixer.py`, `generator.py`
- Test: `tests/unit/test_mixer.py`, `tests/unit/test_scenario.py`

**Interfaces:**
- Consumes: `SyntheticConfig`, schéma GT Task 2.
- Produces :
  - `PiperTts(TtsEngine)` : `synthesize(text: str, voice: str) -> tuple[np.ndarray, int]` (voix fr_FR via hf_hub_download `rhasspy/piper-voices`, cache `models/piper`) ; `AVAILABLE_VOICES: list[str]`.
  - `ScenarioLine(speaker: str, voice: str, text: str)` ; `ScenarioThread(lines: list[ScenarioLine], gaps: list[float])` ; `generate_scenario(n_conversations, speakers_per_thread, n_lines_range, rng) -> list[ScenarioThread]` (textes tirés de banques de phrases FR thématiques distinctes par thread).
  - `place(clips: list[tuple[float, np.ndarray]], length: int) -> np.ndarray` ; `add_noise(mix: np.ndarray, snr_db: float, rng) -> np.ndarray` (bruit blanc filtré passe-bas 3,4 kHz, puissance = signal/10^(snr/10)).
  - `SyntheticGenerator(tts, config).generate(out_dir, seed, n_conversations, speakers_per_thread, n_lines=(4,8)) -> Path` : écrit `mixed.wav` + `ground_truth.json` (schéma conversations) ; timings réels issus des longueurs audio synthétisées.

- [ ] **Step 1:** Tests : SNR exact (±0,5 dB) sur sinus placés + bruit connu ; place additionne sans clip hors bornes ; scénario seedé déterministe (deux appels même seed ⇒ mêmes textes/ordres) ; comptages respectés.

```python
def test_add_noise_exact_snr():
    t = np.arange(16000)/16000; sig = 0.5*np.sin(2*np.pi*220*t)
    noisy = add_noise(sig, snr_db=10.0, rng=np.random.default_rng(0))
    noise = noisy - sig
    snr = 10*np.log10(np.mean(sig**2)/np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=0.5)
```

- [ ] **Step 2:** FAIL → **Step 3:** implémentation (introspection API piper 1.7 au premier appel ; resample voix→16k) → **Step 4:** PASS. Commit : `feat(synthetic): piper tts, seeded scenarios, snr mixer, dataset generator`

### Task 13: Pipeline + CLI + visualisation

**Files:**
- Create: `src/conversation_deconvolution/pipeline.py`, `cli.py`, `conversation/viz.py`
- Test: `tests/unit/test_pipeline_fake.py`

**Interfaces:**
- Consumes: tous les étages.
- Produces :
  - `DeconvolutionPipeline(vad, diarizer, separator, asr, reconstructor, text_embedder, config)` ; `.run(audio: np.ndarray) -> TranscriptResult` ; classement : VAD → diarize → overlaps → pour chaque turn : ASR (contexte ±pad) → Utterances (IoU déjà implicite : 1 utterance par turn) → reconstruct → export.
  - `TranscriptExporter.save(result, path)` ; `plot_timeline(result, path_png, title)` (matplotlib : barres locuteurs, hachures overlaps, couleurs par conversation).
  - CLI typer `app` : `run INPUT -o OUT.json [--config] [--num-speakers N] [--plot PNG]` ; `synth --out DIR [--conversations 2] [--speakers 2] [--seed 0] [--snr-db 15]` ; `evaluate --pred P --gt G` (DER/WER/F1 table rich) ; `viz RESULT WAV --out PNG` ; `benchmark --datasets N --out report.md` (génère N datasets seedés distincts, exécute, agrège moyennes±std en Markdown).

- [ ] **Step 1:** Test pipeline avec composants fake (FakeVad 2 segments, FakeDiarizer 2 locuteurs alternés, FakeAsr textes fixés, HashTextEmbedder) → TranscriptResult cohérent : 2 utterances, ≥1 conversation, JSON exportable et rechargable.

- [ ] **Step 2:** FAIL → **Step 3:** implémentation pipeline/cli/viz → **Step 4:** PASS + `uv run deconvolute --help` OK. Commit : `feat(pipeline,cli): orchestrator, typer cli, timeline viz, benchmark command`

### Task 14: Intégration end-to-end réelle + README quickstart

**Files:**
- Create: `tests/integration/test_e2e_synthetic.py`
- Modify: `README.md` (section Quickstart), `docs/ROADMAP.md` (statuts M1→M5)

**Interfaces:**
- Consumes: tout.

- [ ] **Step 1:** Test marqué `slow` : génère dataset réel (2 conversations × 2 locuteurs, 5 lignes/thread, seed fixe) → `DeconvolutionPipeline` défaut mais `AsrConfig(model_size="tiny")` → run → asserts : JSON valide, nb locuteurs détectés ≥ 2, ≥ 1 conversation, WER moyen < 1,0 (sanité), DER calculé contre GT imprimable.

```python
@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    gt_dir = SyntheticGenerator(PiperTts(), SyntheticConfig()).generate(
        tmp_path/"ds", seed=7, n_conversations=2, speakers_per_thread=2)
    audio = load_audio(gt_dir/"mixed.wav")
    cfg = PipelineConfig.default(); cfg.asr.model_size = "tiny"
    result = build_pipeline(cfg).run(audio)
    speakers = {u.speaker for u in result.utterances}
    assert len(speakers) >= 2 and len(result.conversations) >= 1
```

- [ ] **Step 2:** Exécuter `uv run pytest -m slow tests/integration -q` (téléchargements ~2 Go au premier run) → PASS.
- [ ] **Step 3:** README Quickstart (install uv, synth, run, evaluate, benchmark) ; ROADMAP : M1..M5 cochés avec notes.
- [ ] **Step 4:** Commit : `test(e2e): real-model integration + docs quickstart`

## Self-Review

- Couverture spec : §2 stack ✓ (Tasks 1,10,11,12) · §3 layout ✓ · §4 données ✓ (Task 2) · §5 pipeline ✓ (Tasks 4-6,10,11,13) · §6 synthétique ✓ (Task 12) · §7 évaluation ✓ (Tasks 7-9,13 benchmark) · §8 tests ✓ partout + Task 14 · §9 risques : fallback piper prévu Task 12, seuils configurables Task 3.
- Placeholders : aucun TBD ; chaque algorithme non trivial a son pseudo/code.
- Cohérence types : signatures ci-dessus sont la référence croisée des tasks (Utterance/Segment/VadResult/SpeakerTurn/AsrResult nommés identiquement partout).
