# TSE v2: Per-Frequency FiLM + Contrastive Loss — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve TSE model separation quality by adding per-frequency FiLM conditioning, a stronger embedding MLP, and a contrastive loss that directly optimizes speaker similarity.

**Architecture:** Extend `TseModel` with `FilmBlockV2` (per-frequency γ/β), `ConditioningMLP` (non-linear embedding transform), and a frozen ECAPA embedder for contrastive loss. Config gets 3 new params with defaults. Training loop passes new lambdas to `compute_loss`.

**Tech Stack:** Python 3.12, PyTorch, speechbrain (ECAPA embedder, frozen), existing `TseDataset`/`TseSeparator` unchanged.

**Spec:** `docs/superpowers/specs/2026-09-02-tse-v2-per-frequency-film-contrastive.md`

## Global Constraints

- Python 3.12+, PyTorch, `uv` for dependency management
- STFT params fixed: `n_fft=512`, `hop=256`, `window="hann"`
- ECAPA embedder: `speechbrain/spkrec-ecapa-voxceleb`, 192-d, frozen during TSE training
- `channels=64` default, `freq_bands=32` default
- Param budget: ≤1.5M (target ~715k)
- All existing tests must continue to pass
- No changes to `TseDataset`, `TseSeparator`, `build_pipeline`, CLI, or checkpoint format

---

## Task 1: Add new config params to TseConfig

**Files:**
- Modify: `src/conversation_deconvolution/core/config.py:91-107` (TseConfig)
- Modify: `configs/default.yaml:59-74` (tse section)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TseConfig.freq_bands`, `TseConfig.lambda_rec`, `TseConfig.lambda_sim` — used by Task 2 (model) and Task 3 (training)

- [ ] **Step 1: Add fields to TseConfig**

In `src/conversation_deconvolution/core/config.py`, add three new fields to `TseConfig` after `noise_bandwidth`:

```python
@dataclass
class TseConfig:
    model_path: str = "models/tse/model.pt"
    hparams_path: str = "models/tse/hparams.yaml"
    n_fft: int = 512
    hop: int = 256
    window: str = "hann"
    n_blocks: int = 3
    channels: int = 64
    embed_dim: int = 192
    lr: float = 3e-4
    grad_clip: float = 1.0
    epochs: int = 30
    batch_size: int = 4
    snr_low: float = 10.0
    snr_high: float = 20.0
    noise_bandwidth: float = 3400.0
    freq_bands: int = 32
    lambda_rec: float = 0.5
    lambda_sim: float = 0.5
```

- [ ] **Step 2: Update default.yaml**

In `configs/default.yaml`, add to the `tse:` section:

```yaml
tse:
  model_path: models/tse/model.pt
  hparams_path: models/tse/hparams.yaml
  n_fft: 512
  hop: 256
  window: hann
  n_blocks: 3
  channels: 64
  embed_dim: 192
  lr: 3e-4
  grad_clip: 1.0
  epochs: 30
  batch_size: 4
  snr_low: 10.0
  snr_high: 20.0
  noise_bandwidth: 3400.0
  freq_bands: 32
  lambda_rec: 0.5
  lambda_sim: 0.5
```

- [ ] **Step 3: Add config test**

In `tests/unit/test_config.py`, add a test that the new fields exist and have correct defaults:

```python
def test_tse_config_new_fields():
    from conversation_deconvolution.core.config import TseConfig
    cfg = TseConfig()
    assert cfg.freq_bands == 32
    assert cfg.lambda_rec == 0.5
    assert cfg.lambda_sim == 0.5
```

- [ ] **Step 4: Run config tests**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation_deconvolution/core/config.py configs/default.yaml tests/unit/test_config.py
git commit -m "feat(tse): add freq_bands, lambda_rec, lambda_sim to TseConfig"
```

---

## Task 2: Implement TseModel v2 with per-frequency FiLM and contrastive loss

**Files:**
- Modify: `src/conversation_deconvolution/tse/model.py` (full rewrite of model classes)
- Test: `tests/unit/test_tse_model.py`

**Interfaces:**
- Consumes: `TseConfig.freq_bands`, `TseConfig.lambda_rec`, `TseConfig.lambda_sim` (from Task 1)
- Produces: `TseModel` with updated `forward()`, `compute_loss()` signatures — consumed by Task 3 (training) and `TseSeparator` (inference, unchanged interface)

- [ ] **Step 1: Write failing test for FilmBlockV2 per-frequency shapes**

In `tests/unit/test_tse_model.py`, add:

```python
def test_film_block_v2_per_freq_shapes():
    from conversation_deconvolution.tse.model import FilmBlockV2
    block = FilmBlockV2(channels=64)
    x = torch.randn(2, 64, 257, 50)
    gamma = torch.randn(2, 64, 32, 1)
    beta = torch.randn(2, 64, 32, 1)
    out = block(x, gamma, beta)
    assert out.shape == x.shape
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_tse_model.py::test_film_block_v2_per_freq_shapes -v`
Expected: FAIL (FilmBlockV2 not defined)

- [ ] **Step 3: Implement FilmBlockV2**

In `src/conversation_deconvolution/tse/model.py`, add `FilmBlockV2` after the existing `FilmBlock`:

```python
class FilmBlockV2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, gamma, beta):
        residual = x
        if gamma.shape[2] != x.shape[2]:
            gamma = F.interpolate(gamma, size=(x.shape[2], 1), mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=(x.shape[2], 1), mode="bilinear", align_corners=False)
        out = F.relu(self.bn1(self.conv1(x)))
        out = gamma * out + beta
        out = self.bn2(self.conv2(out))
        out = gamma * out + beta
        return F.relu(out + residual)
```

- [ ] **Step 4: Run FilmBlockV2 test**

Run: `uv run pytest tests/unit/test_tse_model.py::test_film_block_v2_per_freq_shapes -v`
Expected: PASS

- [ ] **Step 5: Write failing test for ConditioningMLP**

```python
def test_conditioning_mlp_output_shape():
    from conversation_deconvolution.tse.model import ConditioningMLP
    mlp = ConditioningMLP(embed_dim=192, channels=64, freq_bands=32)
    emb = torch.randn(2, 192)
    gamma, beta = mlp(emb)
    assert gamma.shape == (2, 64, 32, 1)
    assert beta.shape == (2, 64, 32, 1)
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/unit/test_tse_model.py::test_conditioning_mlp_output_shape -v`
Expected: FAIL (ConditioningMLP not defined)

- [ ] **Step 7: Implement ConditioningMLP**

```python
class ConditioningMLP(nn.Module):
    def __init__(self, embed_dim=192, channels=64, freq_bands=32):
        super().__init__()
        out_dim = channels * freq_bands
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )
        self.channels = channels
        self.freq_bands = freq_bands

    def forward(self, emb):
        out = self.net(emb)
        gamma = out.view(-1, self.channels, self.freq_bands, 1)
        beta = out.view(-1, self.channels, self.freq_bands, 1)
        return gamma, beta
```

- [ ] **Step 8: Run ConditioningMLP test**

Run: `uv run pytest tests/unit/test_tse_model.py::test_conditioning_mlp_output_shape -v`
Expected: PASS

- [ ] **Step 9: Write failing test for TseModel v2 forward shapes**

```python
def test_tse_model_v2_forward_shapes():
    from conversation_deconvolution.tse.model import TseModel
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3, freq_bands=32)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        mask = model(mix, ref_emb)
    assert mask.shape[0] == 1
    assert mask.shape[1] == 257
    assert mask.shape[2] > 0
```

- [ ] **Step 10: Run to verify it fails (TseModel doesn't accept freq_bands yet)**

Run: `uv run pytest tests/unit/test_tse_model.py::test_tse_model_v2_forward_shapes -v`
Expected: FAIL (unexpected keyword argument 'freq_bands')

- [ ] **Step 11: Rewrite TseModel to use new components**

Replace the `TseModel` class in `src/conversation_deconvolution/tse/model.py`:

```python
class TseModel(nn.Module):
    def __init__(
        self, n_fft=512, hop=256, window="hann", n_blocks=3, channels=128,
        embed_dim=192, freq_bands=32,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.window = window
        self.channels = channels
        self.freq_bands = freq_bands

        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([FilmBlockV2(channels) for _ in range(n_blocks)])
        self.cond_conv = nn.Conv2d(channels + embed_dim, channels, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.conditioning = ConditioningMLP(embed_dim, channels, freq_bands)
        self.sigmoid = nn.Sigmoid()
        self._frozen_embedder = None

    def _get_frozen_embedder(self):
        if self._frozen_embedder is None:
            from conversation_deconvolution.diarization.embeddings import EcapaEmbedder
            self._frozen_embedder = EcapaEmbedder()
        return self._frozen_embedder

    def _frozen_embed(self, waveform):
        import numpy as np
        embedder = self._get_frozen_embedder()
        audio_np = waveform.detach().cpu().numpy().astype(np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze(0)
        emb = np.asarray(embedder.embed(audio_np), dtype=np.float64)
        norm = float(np.linalg.norm(emb)) or 1.0
        return torch.from_numpy(emb / norm).float().to(waveform.device).unsqueeze(0)

    def forward(self, mix, ref_emb):
        spec = _stft(mix, self.n_fft, self.hop, self.window)
        real = spec.real.unsqueeze(1)
        imag = spec.imag.unsqueeze(1)
        x = torch.cat([real, imag], dim=1)
        x = self.encoder(x)
        gamma, beta = self.conditioning(ref_emb)
        for block in self.blocks:
            x = block(x, gamma, beta)
        b, _, f, t = x.shape
        emb_tiled = ref_emb.view(b, -1, 1, 1).expand(b, ref_emb.shape[1], f, t)
        x = torch.cat([x, emb_tiled], dim=1)
        x = self.cond_conv(x)
        mask = self.sigmoid(self.decoder(x).squeeze(1))
        return mask

    def apply_mask(self, mix, mask):
        spec = _stft(mix, self.n_fft, self.hop, self.window)
        if mask.is_complex():
            masked_spec = mask * spec
        else:
            masked_spec = mask * spec
        return _istft(masked_spec, mix, self.n_fft, self.hop, self.window)

    def compute_loss(self, mix, target, ref_emb, lambda_rec=0.5, lambda_sim=0.5):
        mask = self.forward(mix, ref_emb)
        est = self.apply_mask(mix, mask)
        rec_loss = -_si_sdr(est, target).mean()
        est_emb = self._frozen_embed(est)
        ref_emb_norm = ref_emb / (ref_emb.norm(dim=-1, keepdim=True) + 1e-8)
        sim_loss = 1 - F.cosine_similarity(est_emb, ref_emb_norm).mean()
        return lambda_rec * rec_loss + lambda_sim * sim_loss
```

- [ ] **Step 12: Run TseModel v2 forward shapes test**

Run: `uv run pytest tests/unit/test_tse_model.py::test_tse_model_v2_forward_shapes -v`
Expected: PASS

- [ ] **Step 13: Update existing tests for new signature**

Update `test_forward_shapes` in `tests/unit/test_tse_model.py` to pass `freq_bands=32`:

```python
def test_forward_shapes():
    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32)
    # ... rest unchanged
```

Update `test_param_count` range:

```python
def test_param_count():
    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32)
    n_params = sum(p.numel() for p in model.parameters())
    assert 500_000 <= n_params <= 1_500_000
```

- [ ] **Step 14: Write param count test for default (channels=64)**

```python
def test_param_count_default():
    from conversation_deconvolution.tse.model import TseModel
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3, freq_bands=32)
    n_params = sum(p.numel() for p in model.parameters())
    assert 500_000 <= n_params <= 1_000_000
```

- [ ] **Step 15: Write contrastive loss test**

```python
def test_contrastive_loss():
    from conversation_deconvolution.tse.model import TseModel
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3, freq_bands=32)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    target = torch.randn(1, 16000)
    with torch.no_grad():
        loss = model.compute_loss(mix, target, ref_emb, lambda_rec=0.5, lambda_sim=0.5)
    assert loss.item() > 0
    assert math.isfinite(loss.item())
```

- [ ] **Step 16: Run all model tests**

Run: `uv run pytest tests/unit/test_tse_model.py -v`
Expected: ALL PASS

- [ ] **Step 17: Commit**

```bash
git add src/conversation_deconvolution/tse/model.py tests/unit/test_tse_model.py
git commit -m "feat(tse): per-frequency FiLM, conditioning MLP, contrastive loss"
```

---

## Task 3: Update training loop to pass new loss params

**Files:**
- Modify: `src/conversation_deconvolution/tse/train.py:11-50` (train_tse_model)
- Modify: `src/conversation_deconvolution/tse/train.py:53-70` (_save_hparams)
- Test: manual run (no unit test for training loop change)

**Interfaces:**
- Consumes: `TseConfig.freq_bands`, `TseConfig.lambda_rec`, `TseConfig.lambda_sim` (from Task 1), `TseModel.compute_loss` new signature (from Task 2)
- Produces: checkpoint + hparams with new fields

- [ ] **Step 1: Update train_tse_model to pass freq_bands and lambdas**

In `src/conversation_deconvolution/tse/train.py`, update the model construction and loss call:

```python
def train_tse_model(dataset, config: TseConfig, out_path: str) -> str:
    model = TseModel(
        n_fft=config.n_fft,
        hop=config.hop,
        window=config.window,
        channels=config.channels,
        embed_dim=config.embed_dim,
        n_blocks=config.n_blocks,
        freq_bands=config.freq_bands,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            mix, target, ref_emb, _ = batch
            if isinstance(mix, torch.Tensor):
                mix = mix.to(device).float()
                target = target.to(device).float()
                ref_emb = ref_emb.to(device).float()
            else:
                mix = torch.from_numpy(np.asarray(mix, dtype=np.float32)).to(device)
                target = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device)
                ref_emb = (
                    torch.from_numpy(np.asarray(ref_emb, dtype=np.float64)).to(device).float()
                )
            optimizer.zero_grad()
            loss = model.compute_loss(
                mix, target, ref_emb,
                lambda_rec=config.lambda_rec,
                lambda_sim=config.lambda_sim,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(loader))
        print(f"epoch {epoch + 1}/{config.epochs} loss={avg:.4f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    _save_hparams(config, out_path)
    return out_path
```

- [ ] **Step 2: Update _save_hparams to include new fields**

```python
def _save_hparams(config: TseConfig, model_path: str) -> None:
    import yaml

    hparams_path = model_path.replace(".pt", ".yaml")
    with open(hparams_path, "w") as f:
        yaml.dump(
            {
                "n_fft": config.n_fft,
                "hop": config.hop,
                "window": config.window,
                "n_blocks": config.n_blocks,
                "channels": config.channels,
                "embed_dim": config.embed_dim,
                "freq_bands": config.freq_bands,
                "lr": config.lr,
                "lambda_rec": config.lambda_rec,
                "lambda_sim": config.lambda_sim,
            },
            f,
            default_flow_style=False,
        )
```

- [ ] **Step 3: Run all unit tests to confirm no regressions**

Run: `uv run pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/conversation_deconvolution/tse/train.py
git commit -m "feat(tse): pass freq_bands and lambda params in training loop"
```

---

## Task 4: Verify end-to-end with lint and typecheck

**Files:**
- No new files

**Interfaces:**
- Consumes: all prior tasks
- Produces: green CI

- [ ] **Step 1: Run linter**

Run: `uv run ruff check src/conversation_deconvolution/tse/ src/conversation_deconvolution/core/config.py`
Expected: no errors

- [ ] **Step 2: Run type checker if configured**

Run: `uv run pyright src/conversation_deconvolution/tse/model.py src/conversation_deconvolution/tse/train.py src/conversation_deconvolution/core/config.py` (or skip if not configured)
Expected: no errors or only pre-existing warnings

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/integration`
Expected: ALL PASS

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(tse): lint/type fixes for v2"
```

---

*End of plan. After execution, train v2 and benchmark:*
```bash
uv run deconvolute train-tse --epochs 15 --out models/tse/model_v2.pt
uv run deconvolute benchmark --datasets 4 --separation-backend tse
```
