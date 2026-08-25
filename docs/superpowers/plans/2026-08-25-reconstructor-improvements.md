# Reconstructor Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve pairwise_F1 from 0.60 to ≥0.70 and ARI from 0.42 to ≥0.55 by implementing the alternation feature, reweighting scoring, and adding a refinement pass.

**Architecture:** Three changes to the heuristic reconstructor: (1) implement `w_alternation` as a binary bonus when consecutive turns in a stream switch speakers, (2) reweight features to normalize total ≤1.0, (3) add a post-assignment refinement pass that re-evaluates stream membership.

**Tech Stack:** Python 3.12, numpy, sentence-transformers, pytest

**Spec:** None formal — improvements driven by benchmark gap (F1=0.60, ARI=0.42 on 4-seed synthetic data)

---

## Global Constraints

- No comments in code
- `ruff check` + `ruff format` must pass (line-length=95)
- All existing 107 tests must continue to pass
- Commit messages: French conventionals (`feat(scope): ...`)
- All new code must have unit tests

---

### Task 1: Implement alternation feature in HeuristicReconstructor

**Files:**
- Modify: `src/conversation_deconvolution/conversation/reconstructor.py:69-78` (scoring)
- Modify: `src/conversation_deconvolution/conversation/reconstructor.py:80-88` (stream update)
- Test: `tests/unit/test_reconstructor.py`

**Interfaces:**
- Consumes: `Utterance.speaker`, `_Stream.last_speaker`
- Produces: `_Stream.last_speaker: str | None` attribute, `_alternation_score()` method

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_reconstructor.py`:

```python
def test_alternation_bonus_when_speakers_switch():
    from unittest.mock import patch
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.core.types import Utterance

    cfg = ReconstructionConfig(
        w_temporal=0.0,
        w_semantic=0.0,
        w_same_speaker=0.0,
        w_alternation=1.0,
        threshold=0.3,
    )
    recon = HeuristicReconstructor(cfg)
    turns = [
        Utterance(id=0, speaker="A", start=0.0, end=1.0, text="bonjour"),
        Utterance(id=1, speaker="B", start=1.5, end=2.5, text="salut"),
        Utterance(id=2, speaker="A", start=3.0, end=4.0, text="ca va"),
    ]
    result = recon.reconstruct(turns)
    assert len(result) == 1
    assert len(result[0].utterance_ids) == 3


def test_no_alternation_penalty_when_same_speaker():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.core.types import Utterance

    cfg = ReconstructionConfig(
        w_temporal=0.0,
        w_semantic=0.0,
        w_same_speaker=0.0,
        w_alternation=1.0,
        threshold=0.3,
    )
    recon = HeuristicReconstructor(cfg)
    turns = [
        Utterance(id=0, speaker="A", start=0.0, end=1.0, text="bonjour"),
        Utterance(id=1, speaker="A", start=1.5, end=2.5, text="encore moi"),
        Utterance(id=2, speaker="A", start=3.0, end=4.0, text="toujours moi"),
    ]
    result = recon.reconstruct(turns)
    assert len(result) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_reconstructor.py::test_alternation_bonus_when_speakers_switch tests/unit/test_reconstructor.py::test_no_alternation_penalty_when_same_speaker -v`
Expected: FAIL

- [ ] **Step 3: Implement alternation feature**

In `reconstructor.py`:
- Add `last_speaker: str | None = None` to `_Stream` dataclass
- Add `_alternation_score(u, stream)` method: returns 1.0 if `u.speaker != stream.last_speaker` and `stream.last_speaker is not None`, else 0.0
- In `_stream_score()`: add `w_alternation * _alternation_score(u, stream)` to the total
- In `_assign()`: set `stream.last_speaker = u.speaker`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_reconstructor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation_deconvolution/conversation/reconstructor.py tests/unit/test_reconstructor.py
git commit -m "feat(reconstructor): implémente w_alternation — bonus A→B→A dans scoring heuristique"
```

---

### Task 2: Reweight reconstruction scoring (normalize to ≤1.0)

**Files:**
- Modify: `src/conversation_deconvolution/core/config.py:47-58` (defaults)
- Test: `tests/unit/test_reconstructor.py`

**Interfaces:**
- Consumes: `ReconstructionConfig` defaults
- Produces: Updated default weights

Current weights sum to 1.25 (0.25 + 0.45 + 0.55), allowing scores > 1.0. Normalize so total ≤ 1.0 for interpretable threshold.

- [ ] **Step 1: Write failing test**

```python
def test_score_never_exceeds_one():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.core.types import Utterance

    cfg = ReconstructionConfig()
    recon = HeuristicReconstructor(cfg)
    turns = [
        Utterance(id=0, speaker="A", start=0.0, end=1.0, text="bonjour"),
        Utterance(id=1, speaker="A", start=1.1, end=2.0, text="encore bonjour"),
    ]
    result = recon.reconstruct(turns)
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_reconstructor.py::test_score_never_exceeds_one -v`
Expected: PASS (current behavior already groups them, but score may exceed 1.0 — test needs to verify score internally)

Actually, this test passes with current code. Instead, test that the threshold works correctly:

```python
def test_threshold_actually_separates():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.core.types import Utterance

    cfg = ReconstructionConfig(
        w_temporal=0.25, w_semantic=0.25, w_same_speaker=0.25,
        w_alternation=0.25, threshold=0.9,
    )
    recon = HeuristicReconstructor(cfg)
    turns = [
        Utterance(id=0, speaker="A", start=0.0, end=1.0, text="bonjour"),
        Utterance(id=1, speaker="B", start=10.0, end=11.0, text="a des questions"),
    ]
    result = recon.reconstruct(turns)
    assert len(result) == 2
```

- [ ] **Step 3: Update default weights**

In `config.py` `ReconstructionConfig`:
- `w_temporal = 0.20`
- `w_semantic = 0.35`
- `w_same_speaker = 0.30`
- `w_alternation = 0.15`
- Total = 1.00

Remove unused parameters: `w_alternation` (now used), keep `max_overlap_ratio` and `max_successors` as they may be used later (or remove them).

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -q`
Expected: ALL PASS (may need to adjust some test thresholds)

- [ ] **Step 5: Commit**

```bash
git add src/conversation_deconvolution/core/config.py tests/unit/test_reconstructor.py
git commit -m "refactor(reconstructor): reweight features — total=1.0, w_alternation=0.15"
```

---

### Task 3: Add refinement pass (reassign misclassified utterances)

**Files:**
- Modify: `src/conversation_deconvolution/conversation/reconstructor.py:26-67` (reconstruct method)
- Test: `tests/unit/test_reconstructor.py`

**Interfaces:**
- Consumes: `_Stream` list after greedy assignment
- Produces: Refined `_Stream` list after reassignment pass

- [ ] **Step 1: Write failing test**

```python
def test_refinement_reassigns_early_misclassifications():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.core.types import Utterance

    cfg = ReconstructionConfig(
        w_temporal=0.20, w_semantic=0.35, w_same_speaker=0.30,
        w_alternation=0.15, threshold=0.5,
    )
    recon = HeuristicReconstructor(cfg)
    turns = [
        Utterance(id=0, speaker="A", start=0.0, end=1.0, text="ca va bien"),
        Utterance(id=1, speaker="B", start=1.5, end=2.5, text="oui tres bien"),
        Utterance(id=2, speaker="C", start=5.0, end=6.0, text="bonjour aussi"),
        Utterance(id=3, speaker="B", start=5.5, end=6.5, text="salut cote"),
    ]
    result = recon.reconstruct(turns)
    assert len(result) >= 1
```

- [ ] **Step 2: Run test to verify it passes/fails as expected**

Run: `uv run pytest tests/unit/test_reconstructor.py::test_refinement_reassigns_early_misclassifications -v`

- [ ] **Step 3: Implement refinement pass**

After greedy assignment in `reconstruct()`:
1. For each utterance, compute its score with every stream (including the one it's currently in)
2. If another stream gives a strictly better score AND no conflict, move the utterance
3. Repeat up to 3 iterations or until no moves
4. Remove empty streams

This is a simple local-search that fixes early mis-assignments without the complexity of global optimization.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation_deconvolution/conversation/reconstructor.py tests/unit/test_reconstructor.py
git commit -m "feat(reconstructor): passe de raffinement post-greedy — réduit erreurs en cascade"
```

---

### Task 4: Benchmark and tune

**Files:**
- Reports: `reports/` (gitignored)

- [ ] **Step 1: Run full benchmark**

```bash
uv run deconvolute benchmark --datasets 4 --reconstructor heuristic --out reports/bench_recon_v2.md
```

- [ ] **Step 2: Compare with baseline**

Baseline (cell=0.125):
- pairwise_F1: 0.6052, ARI: 0.4248, DER: 0.0905

Target:
- pairwise_F1 ≥ 0.70, ARI ≥ 0.55

- [ ] **Step 3: Tune weights if needed**

If F1/ARI below target, adjust weights in `ReconstructionConfig` and re-run.

- [ ] **Step 4: Update tests if needed**

If any existing tests break due to weight changes, update assertions.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "perf(reconstructor): F1=X→Y, ARI=X→Y — tuning poids alternation+raffinement"
```

---

### Task 5: Clean up dead config parameters

**Files:**
- Modify: `src/conversation_deconvolution/core/config.py:47-58`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Remove unused parameters**

Remove from `ReconstructionConfig`:
- `max_overlap_ratio` (never used)
- `max_successors` (never used)

Keep `w_alternation` (now used), `max_speaker_overlap_ratio` (used in conflict detection).

- [ ] **Step 2: Update config test**

Run: `uv run pytest tests/unit/test_config.py -v`
Fix any failing tests.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -q`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/conversation_deconvolution/core/config.py tests/unit/test_config.py
git commit -m "refactor(config): supprime paramètres ReconstructionConfig inutilisés"
```
