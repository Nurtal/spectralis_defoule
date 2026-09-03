# Spec — TSE v2: Per-Frequency FiLM + Contrastive Loss

## 1. Problem

After ADR-0011, the TSE model (STFT-FiLM, ~339k params, channels=64) is validated
marginally: WER overlap TSE 0.776 < OFF 0.803 (+0.027) but F1/ARI unchanged
(0.651/0.480). Investigation reveals:

| Symptom | Value | Oracle |
|---|---|---|
| Mask diff | 0.01–0.03 | — |
| Est vs ref sim | ~0.05 | 0.31–0.47 |
| Stems corr | 0.99 | — |
| SI-SDR | 35–60 dB | — |

**Root causes identified:**
1. FiLM conditioning is global (one γ/β per channel, shape `(B, C, 1, 1)`) — cannot
   learn frequency-dependent speaker signatures.
2. Conditioning network is a single linear layer (192→64) — too weak to transform
   ECAPA embeddings into effective modulation signals.
3. SI-SDR loss optimizes waveform fidelity but does not enforce speaker similarity —
   model converges to a safe average mask.

## 2. Approach

Three targeted improvements to the existing TseModel architecture, preserving the
STFT-FiLM backbone, training pipeline, dataset, CLI, and inference interface.

### 2.1 Per-frequency FiLM

Current: `gamma_fc = Linear(embed_dim, channels)` → gamma shape `(B, C, 1, 1)` — same
modulation for all frequency bins.

New: `gamma_fc = Linear(embed_dim, channels × freq_bands)` → reshape to
`(B, C, freq_bands, 1)` → interpolate to `(B, C, F, 1)` where `F = n_fft // 2 + 1 = 257`.

This allows the model to learn that, e.g., speaker A's energy concentrates in 300–800 Hz
while speaker B's is in 1–3 kHz, applying different gain/offset per frequency.

`freq_bands` is a hyperparameter (default 32) that controls the frequency resolution of
the FiLM modulation before interpolation. This keeps parameter count manageable while
providing meaningful frequency selectivity.

### 2.2 Conditioning MLP

Current: `gamma_fc = Linear(192, 64)`, `beta_fc = Linear(192, 64)`.

New: `gamma_fc = Sequential(Linear(192, 128), ReLU, Linear(128, channels × freq_bands))`,
same for beta_fc. Non-linear transform of the ECAPA embedding before producing
modulation parameters.

Adds ~128 × 128 = 16k params per projection (negligible relative to per-frequency
expansion).

### 2.3 Contrastive loss

Keep SI-SDR as reconstruction loss (λ_rec default 0.5).

Add cosine similarity loss (λ_sim default 0.5):
- After producing the estimated signal `est = apply_mask(mix, mask)`, extract its
  ECAPA embedding via a **frozen** embedder (same `EcapaEmbedder` used in dataset).
- Compute `cos_sim = cosine_similarity(est_emb, ref_emb)`.
- Loss contribution: `1 - cos_sim` (minimize negative similarity).
- Total loss: `λ_rec × (-si_sdr) + λ_sim × (1 - cos_sim)`.

This directly optimizes the metric that matters most (est vs ref sim, currently ~0.05
vs oracle 0.31–0.47).

**Frozen embedder:** loaded lazily on first `compute_loss` call, parameters frozen,
device跟随 model device. No gradient flows through the embedder.

## 3. Architecture detail

```
TseModel v2
├── STFT (n_fft=512, hop=256, hann)
├── Encoder: Conv2d(2→C) + Conv2d(C→C) + BN + ReLU ×2
├── FiLM blocks × n_blocks (default 3):
│   ├── FilmBlockV2:
│   │   ├── Conv2d(C→C, 3×3) + BN + ReLU
│   │   ├── γ·x + β  (per-frequency: γ,β shape B×C×F×1)
│   │   ├── Conv2d(C→C, 3×3) + BN
│   │   ├── γ·out + β
│   │   └── ReLU + residual
│   └── gamma, beta from ConditioningMLP
├── ConditioningMLP:
│   ├── Linear(192→128) + ReLU + Linear(128 → C × freq_bands)
│   └── reshape → (B, C, freq_bands, 1) → interpolate to (B, C, F, 1)
├── cond_conv: Conv2d(C + embed_dim → C, 1×1)  [unchanged]
├── Decoder: Conv2d(C→C) + BN + ReLU + Conv2d(C→1) + Sigmoid
└── apply_mask: mask × STFT → ISTFT
```

### compute_loss v2

```python
def compute_loss(self, mix, target, ref_emb, lambda_rec=0.5, lambda_sim=0.5):
    mask = self.forward(mix, ref_emb)
    est = self.apply_mask(mix, mask)
    rec_loss = -_si_sdr(est, target).mean()
    est_emb = self._frozen_embed(est)
    sim_loss = 1 - F.cosine_similarity(est_emb, ref_emb).mean()
    return lambda_rec * rec_loss + lambda_sim * sim_loss
```

## 4. Hyperparameters (TseConfig additions)

| Param | Default | Description |
|---|---|---|
| `freq_bands` | 32 | FiLM frequency resolution before interpolation |
| `lambda_rec` | 0.5 | Weight for SI-SDR reconstruction loss |
| `lambda_sim` | 0.5 | Weight for cosine similarity contrastive loss |

Existing params unchanged: `n_fft=512`, `hop=256`, `n_blocks=3`, `channels=64`,
`embed_dim=192`, `lr=3e-4`, `grad_clip=1.0`, `epochs=30`, `batch_size=4`.

## 5. Param count estimate

| Component | v1 (channels=64) | v2 (channels=64, freq_bands=32) |
|---|---|---|
| Encoder | ~25k | ~25k |
| FiLM blocks | ~111k | ~111k |
| gamma_fc | 12k | 128×128 + 128×(64×32) = 264k |
| beta_fc | 12k | 264k |
| cond_conv | ~13k | ~13k |
| Decoder | ~38k | ~38k |
| **Total** | **~339k** | **~715k** |

Under 1M params, well within the ≤4M constraint from the TSE spec.

## 6. Files touched

| File | Change |
|---|---|
| `tse/model.py` | `FilmBlockV2` (per-freq FiLM), `ConditioningMLP`, `_frozen_embed`, `compute_loss` v2, `TseModel` updated |
| `core/config.py` | Add `freq_bands: int = 32`, `lambda_rec: float = 0.5`, `lambda_sim: float = 0.5` to `TseConfig` |
| `configs/default.yaml` | Add `freq_bands: 32`, `lambda_rec: 0.5`, `lambda_sim: 0.5` to `tse:` section |
| `tse/train.py` | Pass new config params to `compute_loss`, pass lambda values |
| `tests/unit/test_tse_model.py` | Update param count range, add contrastive loss test, add per-freq shape test |

## 7. What stays the same

- STFT parameters (n_fft=512, hop=256, hann window)
- Encoder/decoder conv structure (2-layer each)
- cond_conv (embedding tiled, concatenated, 1×1 conv)
- Training loop (`train_tse_model`)
- Dataset (`TseDataset`) — no changes needed
- CLI commands (`train-tse`, `run --separation-backend tse`, `benchmark`)
- Inference interface (`TseSeparator.separate`)
- `apply_mask` (magnitude mask, sigmoid → ISTFT)
- `build_pipeline` factory (no changes)
- Checkpoint format (`model.pt` + `hparams.yaml`)

## 8. Backward compatibility

- New config params have defaults → existing configs work without modification
- New model has different state_dict keys → old checkpoints won't load (intentional,
  architecture change). `build_pipeline` already handles load failure gracefully
  (try/except pass).
- `compute_loss` signature changes but is only called from `train.py` (internal).

## 9. Testing

| Test | File | What |
|---|---|---|
| `test_forward_shapes_v2` | `test_tse_model.py` | mask shape unchanged (B, 257, T) |
| `test_per_freq_film_shapes` | `test_tse_model.py` | gamma/beta have freq dimension |
| `test_param_count_v2` | `test_tse_model.py` | 500k ≤ params ≤ 1.5M |
| `test_contrastive_loss` | `test_tse_model.py` | loss is finite, > 0, has grad |
| `test_deterministic_v2` | `test_tse_model.py` | same input → same output |
| `test_mask_bounded_v2` | `test_tse_model.py` | mask ∈ [0, 1] |
| Existing unit tests | `test_tse_*.py` | All still pass |

## 10. Evaluation plan

After implementation:
1. Run existing unit tests to confirm no regressions.
2. Train v2 model: `uv run deconvolute train-tse --epochs 15 --out models/tse/model_v2.pt`
3. Benchmark: `uv run deconvolute benchmark --datasets 4 --separation-backend tse`
4. Compare against v1 baseline (WER overlap, est vs ref sim, mask diff, stems corr).
5. Target: est vs ref sim > 0.15 (from ~0.05), WER overlap < 0.776 (from 0.776).

---

*Spec written. Prochaine étape : révision utilisateur, puis invocation de writing-plans.*
