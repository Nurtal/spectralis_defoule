# TSE — Target Speaker Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un backend de séparation TSE (STFT-FiLM conditionné par embeddings ECAPA) au pipeline, entraînable sur les datasets `train_3000_*` existants, avec critères SI-SDR ≥ 5 dB et WER overlap < 0,807.

**Architecture:** Nouveau package `tse/` (modèle, dataset, entraînement) + nouveau séparateur `separation/tse_separator.py` intégré via `build_pipeline`. Réutilise les datasets `train_3000_*` existants (4 speakers, Piper FR, cache TTS actif).

**Tech Stack:** Python 3.12, uv, torch cu124, torchaudio, numpy, speechbrain ECAPA (cache existant), piper-tts (cache existant), pytest, ruff.

**Spec :** `docs/superpowers/specs/2026-08-31-tse-target-speaker-extraction-design.md`

## Global Constraints

- Python 3.12, gestion uv (`uv sync` doit suffire sur checkout neuf).
- Torch depuis l'index `https://download.pytorch.org/whl/cu124` (ne pas toucher aux sections `[tool.uv]`).
- Audio canonique interne : mono, 16000 Hz, float32, plage [-1, 1].
- Aucune dépendance gated HF ; aucun appel cloud.
- Pas de commentaires dans le code (style repo) ; docstrings une ligne autorisées.
- Tests unitaires < 10 s sans téléchargement ; intégration sous marqueur `slow`.
- Chaque tâche se termine par un commit vert (`make lint test` passe).
- ruff line-length 95, `ruff format --check` propre.

---

### Task 1: TseConfig + config.py + default.yaml

**Files:**
- Modify: `src/conversation_deconvolution/core/config.py`
- Modify: `configs/default.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces : `TseConfig` dataclass. `PipelineConfig.tse: TseConfig`. `PipelineConfig.from_yaml` accepte `tse:`.
- `configs/default.yaml` section `tse:`.

- [ ] **Step 1: Write failing test**

Ajouter à `tests/unit/test_config.py` :

```python
def test_tse_section_loads(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("tse:\n  model_path: models/tse/model.pt\n  lr: 1e-4\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.tse.model_path == "models/tse/model.pt"
    assert cfg.tse.lr == 1e-4


def test_tse_defaults():
    cfg = PipelineConfig.default()
    assert cfg.tse.n_fft == 512
    assert cfg.tse.lr == 3e-4
    assert cfg.tse.epochs == 30
    assert cfg.tse.model_path == "models/tse/model.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_tse_section_loads tests/unit/test_config.py::test_tse_defaults -v`
Expected: FAIL (`TseConfig` inconnue / champ absent).

- [ ] **Step 3: Implement TseConfig in config.py**

Dans `src/conversation_deconvolution/core/config.py`, ajouter après `GraphConfig` :

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
```

Ajouter `tse: TseConfig = field(default_factory=TseConfig)` aux champs de `PipelineConfig`.
Ajouter `"tse": TseConfig` au dict `sections` dans `from_yaml`.

- [ ] **Step 4: Add tse section to configs/default.yaml**

Ajouter après la section `graph:` dans `configs/default.yaml` :

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
```

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/unit/test_config.py -q && make lint`
Expected: PASS.

Commit : `git add src/conversation_deconvolution/core/config.py configs/default.yaml tests/unit/test_config.py`
`git commit -m "feat(config): section tse pour Target Speaker Extraction"`

---

### Task 2: TSE Model (`tse/model.py`)

**Files:**
- Create: `src/conversation_deconvolution/tse/__init__.py`
- Create: `src/conversation_deconvolution/tse/model.py`
- Test: `tests/unit/test_tse_model.py`

**Interfaces:**
- Produces : `TseModel(nn.Module)` — STFT-FiLM, ~3M params, loss SI-SDR.
- `forward(mix, ref_emb) -> mask` (shape: [B, n_frames, n_freq]).
- `compute_loss(mix, target, ref_emb) -> si_sdr` (scalar).

- [ ] **Step 1: Write failing tests**

Créer `tests/unit/test_tse_model.py` :

```python
import torch
import pytest

from conversation_deconvolution.tse.model import TseModel


def test_forward_shapes():
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        mask = model(mix, ref_emb)
    n_frames = 1 + (16000 - 512) // 256
    assert mask.shape == (1, n_frames, 257)


def test_loss_reduces():
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    target = torch.randn(1, 16000)
    with torch.no_grad():
        loss = model.compute_loss(mix, target, ref_emb)
    assert loss.item() > 0
    assert torch.isfinite(loss.item())


def test_param_count_under_4m():
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 4_000_000
    assert n_params >= 1_000_000


def test_deterministic():
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        out1 = model(mix, ref_emb)
        out2 = model(mix, ref_emb)
    assert torch.allclose(out1, out2)


def test_mask_bounded():
    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        mask = model(mix, ref_emb)
    assert mask.min() >= 0 and mask.max() <= 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_tse_model.py -v`
Expected: FAIL (`TseModel` not defined).

- [ ] **Step 3: Implement TseModel**

Créer `src/conversation_deconvolution/tse/model.py` :

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _stft(mix, n_fft=512, hop=256, window="hann"):
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    win = torch.hann_window(n_fft, device=mix.device) if window == "hann" else torch.kaiser_window(n_fft, device=mix.device)
    spec = torch.stft(mix, n_fft=n_fft, hop_length=hop, win=win, return_complex=True, normalized=False)
    return spec


def _istft(mask_spec, mix, n_fft=512, hop=256, window="hann"):
    win = torch.hann_window(n_fft, device=mask_spec.device) if window == "hann" else torch.kaiser_window(n_fft, device=mask_spec.device)
    return torch.istft(mask_spec * _stft(mix, n_fft=n_fft, hop=hop, window=window), n_fft=n_fft, hop_length=hop, win=win, length=mix.shape[-1])


class FilmBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, gamma, beta):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = gamma * out + beta
        out = self.bn2(self.conv2(out))
        out = gamma * out + beta
        return F.relu(out + residual)


class TseModel(nn.Module):
    def __init__(self, n_fft=512, hop=256, window="hann", n_blocks=3, channels=64, embed_dim=192):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.window = window
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([FilmBlock(channels) for _ in range(n_blocks)])
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        self.gamma_fc = nn.Sequential(nn.Linear(embed_dim, channels), nn.Sigmoid())
        self.beta_fc = nn.Sequential(nn.Linear(embed_dim, channels), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, mix, ref_emb):
        spec = _stft(mix, self.n_fft, self.hop, self.window)
        real = spec.real
        imag = spec.imag
        x = torch.cat([real, imag], dim=1)
        x = self.encoder(x)
        gamma = self.gamma_fc(ref_emb).view(-1, self.channels, 1, 1)
        beta = self.beta_fc(ref_emb).view(-1, self.channels, 1, 1)
        for block in self.blocks:
            x = block(x, gamma, beta)
        mask_real, mask_imag = self.decoder(x).chunk(2, dim=1)
        mask_spec = torch.complex(mask_real, mask_imag)
        return self.sigmoid(mask_spec)

    def compute_loss(self, mix, target, ref_emb):
        mask = self.forward(mix, ref_emb)
        spec = _stft(mix, self.n_fft, self.hop, self.window)
        est_spec = mask * spec
        est = torch.istft(est_spec, n_fft=self.n_fft, hop_length=self.hop, length=target.shape[-1])
        return _si_sdr(est, target)


def _si_sdr(estimate, reference):
    eps = 1e-8
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(reference * estimate, dim=-1, keepdim=True)
    alpha = proj / ref_energy
    residual = estimate - alpha * reference
    return 10 * torch.log10(ref_energy / (torch.sum(residual ** 2, dim=-1, keepdim=True) + eps))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tse_model.py -v`
Expected: ALL PASS. Fix any shape/type issues.

- [ ] **Step 5: Commit**

```bash
git add src/conversation_deconvolution/tse/ tests/unit/test_tse_model.py
git commit -m "feat(tse): STFT-FiLM model ~3M params + SI-SDR loss"
```

---

### Task 3: TSE Dataset (`tse/dataset.py`)

**Files:**
- Create: `src/conversation_deconvolution/tse/dataset.py`
- Test: `tests/unit/test_tse_dataset.py`

**Interfaces:**
- Produits : `TseDataset` — `__len__`, `__getitem__` → `(mix, target, ref_emb, num_speakers)`.
- Utilise les datasets `train_3000_*` existants (ground_truth.json).
- Batch : K speakers, mix + bruit band-limité, cible isolée, embedding référence exclusif.

- [ ] **Step 1: Write failing test**

Créer `tests/unit/test_tse_dataset.py` :

```python
import numpy as np
import pytest

from conversation_deconvolution.tse.dataset import TseDataset
from conversation_deconvolution.synthetic.tts import PiperTts
from conversation_deconvolution.core.config import TseConfig


def test_batch_shapes():
    tts = PiperTts()
    cfg = TseConfig(batch_size=2, snr_low=20.0, snr_high=20.0, noise_bandwidth=0.0)
    ds = TseDataset(tts, cfg, ["data/synthetic/train_3000_0"])
    assert len(ds) > 0
    mix, target, ref_emb, n_spk = ds[0]
    assert mix.shape[0] == 16000
    assert target.shape == mix.shape
    assert ref_emb.shape[1] == 192
    assert 2 <= n_spk <= 4


def test_deterministic():
    tts = PiperTts()
    cfg = TseConfig(batch_size=2, snr_low=20.0, snr_high=20.0, noise_bandwidth=0.0)
    ds1 = TseDataset(tts, cfg, ["data/synthetic/train_3000_0"])
    ds2 = TseDataset(tts, cfg, ["data/synthetic/train_3000_0"])
    b1 = ds1[0]
    b2 = ds2[0]
    assert np.allclose(b1[0], b2[0])
    assert np.allclose(b1[1], b2[1])
    assert np.allclose(b1[2], b2[2])


def test_mix_approximates_target():
    tts = PiperTts()
    cfg = TseConfig(batch_size=2, snr_low=30.0, snr_high=30.0, noise_bandwidth=0.0)
    ds = TseDataset(tts, cfg, ["data/synthetic/train_3000_0"])
    mix, target, _, _ = ds[0]
    diff = np.mean(np.abs(mix - target))
    assert diff < 0.05
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_tse_dataset.py -v`
Expected: FAIL (`TseDataset` not defined).

- [ ] **Step 3: Implement TseDataset**

Créer `src/conversation_deconvolution/tse/dataset.py` :

```python
import json
import numpy as np
from pathlib import Path

from conversation_deconvolution.synthetic.tts import PiperTts
from conversation_deconvolution.core.config import TseConfig


def _bandlimit_noise(noise, bandwidth_hz, sr=16000):
    if bandwidth_hz <= 0:
        return noise
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    low = bandwidth_hz / nyq
    b, a = butter(4, low, btype="low")
    return filtfilt(b, a, noise)


class TseDataset:
    def __init__(self, tts, config, dataset_dirs):
        self.tts = tts
        self.config = config
        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self._utterances = []
        self._load_utterances()

    def _load_utterances(self):
        for d in self.dataset_dirs:
            gt_path = d / "ground_truth.json"
            if not gt_path.exists():
                continue
            data = json.loads(gt_path.read_text())
            for conv in data["conversations"]:
                for u in conv["utterances"]:
                    self._utterances.append({
                        "id": u["id"], "speaker": u["speaker"],
                        "start": u["start"], "end": u["end"],
                        "text": u["text"],
                    })

    def __len__(self):
        return max(1, len(self._utterances) // max(1, self.config.batch_size))

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        speakers = sorted({u["speaker"] for u in self._utterances})
        k = rng.integers(2, min(5, len(speakers)) + 1)
        chosen_speakers = rng.choice(speakers, size=k, replace=False)
        selected = [u for u in self._utterances if u["speaker"] in chosen_speakers]
        rng.shuffle(selected)
        selected = selected[:k]

        sr = 16000
        dur = 1.0
        mix = np.zeros(int(dur * sr), dtype=np.float32)
        target = np.zeros(int(dur * sr), dtype=np.float32)
        target_idx = rng.integers(k)
        target_speaker = chosen_speakers[target_idx]
        ref_utt = None

        for i, spk in enumerate(chosen_speakers):
            spk_utt = [u for u in selected if u["speaker"] == spk]
            if not spk_utt:
                continue
            if spk == target_speaker and len(spk_utt) > 1:
                mix_utt, ref_utt = spk_utt[0], spk_utt[1]
            elif spk == target_speaker:
                mix_utt, ref_utt = spk_utt[0], spk_utt[0]
            else:
                mix_utt, ref_utt = spk_utt[0], None

            audio, _ = self.tts.synthesize(mix_utt["text"], mix_utt["speaker"])
            audio = np.asarray(audio, dtype=np.float32)
            seg_dur = mix_utt["end"] - mix_utt["start"]
            start_s = float(rng.uniform(0, max(0, dur - seg_dur)))
            s = int(start_s * sr)
            e = min(s + int(seg_dur * sr), len(mix))
            seg_len = e - s
            if seg_len > 0 and len(audio) >= seg_len:
                mix[s:e] += audio[:seg_len]
                if spk == target_speaker:
                    target[s:e] += audio[:seg_len]

        if ref_utt is None:
            ref_utt = selected[0]
        ref_audio, _ = self.tts.synthesize(ref_utt["text"], ref_utt["speaker"])
        ref_audio = np.asarray(ref_audio, dtype=np.float32)
        ref_emb = self._compute_embedding(ref_audio)

        noise = rng.randn(len(mix)).astype(np.float32)
        noise = _bandlimit_noise(noise, self.config.noise_bandwidth, sr)
        signal_power = float(np.mean(target ** 2)) + 1e-8
        noise_power = float(np.mean(noise ** 2)) + 1e-8
        scale = np.sqrt(signal_power / noise_power * 10 ** (-self.config.snr_low / 10))
        noise = noise * scale
        mix = mix + noise

        return mix.astype(np.float32), target.astype(np.float32), ref_emb, len(chosen_speakers)

    def _compute_embedding(self, audio):
        from conversation_deconvolution.diarization.embeddings import EcapaEmbedder
        embedder = EcapaEmbedder()
        emb = np.asarray(embedder.embed(audio), dtype=np.float64)
        norm = float(np.linalg.norm(emb)) or 1.0
        return emb / norm

    @classmethod
    def from_existing(cls, tts, config, dataset_dir):
        return cls(tts, config, [dataset_dir])
```

- [ ] **Step 4: Run tests and fix**

Run: `uv run pytest tests/unit/test_tse_dataset.py -v`
Expected: ALL PASS. Fix shapes, API issues, EcapaEmbedder integration.

- [ ] **Step 5: Commit**

```bash
git add src/conversation_deconvolution/tse/dataset.py tests/unit/test_tse_dataset.py
git commit -m "feat(tse): dataset en ligne sur train_3000_*, batch K speakers"
```

---

### Task 4: TSE Separator (`separation/tse_separator.py`)

**Files:**
- Create: `src/conversation_deconvolution/separation/tse_separator.py`
- Modify: `src/conversation_deconvolution/separation/passthrough.py` (ajouter `speaker_refs`)
- Modify: `src/conversation_deconvolution/separation/sepformer.py` (ajouter `speaker_refs`)
- Modify: `src/conversation_deconvolution/core/types.py` (ajouter `meta` à `SeparationResult`)
- Test: `tests/unit/test_tse_separator.py`

**Interfaces:**
- Produits : `TseSeparator` — `separate(mix, regions, speaker_refs=None) -> SeparationResult`.
- `speaker_refs = {label: centroid}` → N stems, indexés par ordre trié des clés.
- Fallback mix si référence dégénérée ou similarité insuffisante.

- [ ] **Step 1: Write failing tests**

Créer `tests/unit/test_tse_separator.py` :

```python
import numpy as np
import pytest

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.core.types import Segment, SeparationResult
from conversation_deconvolution.tse.model import TseModel
from conversation_deconvolution.separation.tse_separator import TseSeparator


def _fake_region():
    return Segment(0.0, 1.0)


def test_separator_returns_stems():
    cfg = TseConfig()
    model = TseModel()
    model.eval()
    sep = TseSeparator(cfg, model)
    mix = np.zeros(16000, dtype=np.float32)
    refs = {"A": np.random.randn(192).astype(np.float64)}
    with torch.no_grad():
        result = sep.separate(mix, [_fake_region()], refs)
    assert len(result.regions) == 1
    assert len(result.regions[0].stems) == 1


def test_no_speaker_refs_returns_empty_stems():
    cfg = TseConfig()
    model = TseModel()
    model.eval()
    sep = TseSeparator(cfg, model)
    mix = np.zeros(16000, dtype=np.float32)
    result = sep.separate(mix, [_fake_region()])
    assert len(result.regions) == 1
    assert len(result.regions[0].stems) == 0


def test_several_speakers():
    cfg = TseConfig()
    model = TseModel()
    model.eval()
    sep = TseSeparator(cfg, model)
    mix = np.zeros(16000, dtype=np.float32)
    refs = {"A": np.random.randn(192).astype(np.float64), "B": np.random.randn(192).astype(np.float64)}
    result = sep.separate(mix, [_fake_region()], refs)
    assert len(result.regions[0].stems) == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_tse_separator.py -v`
Expected: FAIL (`TseSeparator` not defined).

- [ ] **Step 3: Update types.py — add meta to SeparationResult**

Dans `src/conversation_deconvolution/core/types.py`, ajouter `meta` à `SeparationResult` :

```python
@dataclass
class SeparationResult:
    mix: np.ndarray
    regions: list[SeparatedRegion] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def sources(self) -> list[np.ndarray]:
        return [self.mix]
```

- [ ] **Step 4: Implement TseSeparator + update other separators**

Créer `src/conversation_deconvolution/separation/tse_separator.py` :

```python
import torch
import numpy as np

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.core.types import Segment, SeparationResult, SeparatedRegion
from conversation_deconvolution.tse.model import TseModel, _stft, _istft


class TseSeparator:
    def __init__(self, config: TseConfig, model: TseModel):
        self.cfg = config
        self.model = model
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def separate(self, mix, regions, speaker_refs=None):
        mix = np.asarray(mix, dtype=np.float32)
        sr = 16000
        if not speaker_refs:
            return SeparationResult(mix=mix.copy(), regions=[SeparatedRegion(segment=r, stems=[]) for r in regions])

        keys = sorted(speaker_refs.keys())
        ref_embs = []
        for k in keys:
            emb = np.asarray(speaker_refs[k], dtype=np.float64)
            norm = float(np.linalg.norm(emb)) or 1.0
            ref_embs.append(torch.from_numpy(emb / norm).float().to(self.device))
        ref_embs_tensor = torch.stack(ref_embs)

        with torch.no_grad():
            mix_tensor = torch.from_numpy(mix).to(self.device).unsqueeze(0)
            all_stems = [[] for _ in regions]
            for i, region in enumerate(regions):
                s = max(0, int(region.start * sr))
                e = min(len(mix), int(region.end * sr))
                seg = mix_tensor[:, s:e]
                if seg.shape[-1] < self.cfg.n_fft:
                    continue
                for emb in ref_embs_tensor:
                    mask = self.model(seg, emb)
                    spec = _stft(seg.squeeze(0), self.cfg.n_fft, self.cfg.hop, self.cfg.window)
                    est_spec = mask[0] * spec
                    stem = torch.istft(est_spec, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop, length=seg.shape[-1])
                    all_stems[i].append(stem.cpu().numpy().astype(np.float32))

        regions_out = [SeparatedRegion(segment=r, stems=s) for r, s in zip(regions, all_stems)]
        return SeparationResult(mix=mix.copy(), regions=regions_out, meta={"num_speakers": len(keys)})
```

Mettre à jour `PassthroughSeparator.separate` pour accepter `speaker_refs=None` :

```python
class PassthroughSeparator(Separator):
    def separate(self, mix: np.ndarray, regions: list[Segment], speaker_refs=None) -> SeparationResult:
        return SeparationResult(mix=np.asarray(mix, dtype=np.float32).copy())
```

Mettre à jour `SepformerSeparator.separate` pour accepter `speaker_refs=None` (même signature, l'ignorer).

Mettre à jour la classe `Separator` de base dans `passthrough.py` :

```python
class Separator:
    def separate(self, mix: np.ndarray, regions: list[Segment], speaker_refs=None) -> SeparationResult:
        raise NotImplementedError
```

- [ ] **Step 5: Update pipeline.py to pass speaker_refs**

Dans `src/conversation_deconvolution/pipeline.py`, méthode `run` :

```python
    def run(self, audio: np.ndarray) -> TranscriptResult:
        ...
        speaker_refs = self._build_speaker_refs()
        sep_result = self.separator.separate(audio, overlaps, speaker_refs=speaker_refs)
        ...

    def _build_speaker_refs(self):
        centroids = getattr(self.diarizer, "speaker_centroids_", None)
        if not centroids:
            return None
        return {str(k): v for k, v in centroids.items()}
```

Et dans `build_pipeline`, ajouter le backend tse :

```python
    if config.separation.backend == "tse":
        from conversation_deconvolution.separation.tse_separator import TseSeparator
        from conversation_deconvolution.tse.model import TseModel

        tse_cfg = config.tse
        model = TseModel(n_fft=tse_cfg.n_fft, hop=tse_cfg.hop, channels=tse_cfg.channels, embed_dim=tse_cfg.embed_dim, n_blocks=tse_cfg.n_blocks)
        model.load_state_dict(torch.load(tse_cfg.model_path, map_location="cpu"), strict=False)
        model.eval()
        separator = TseSeparator(tse_cfg, model)
    elif config.separation.enabled:
        separator = SepformerSeparator(config.separation)
    else:
        separator = PassthroughSeparator()
```

Ajouter `import torch` en haut de `pipeline.py`.

- [ ] **Step 6: Run all tests and commit**

Run: `uv run pytest -q && make lint`
Expected: PASS.

Commit : `git add src/conversation_deconvolution/core/types.py src/conversation_deconvolution/pipeline.py src/conversation_deconvolution/separation/ src/conversation_deconvolution/tse/ tests/unit/test_tse_separator.py`
`git commit -m "feat(separation): TseSeparator + speaker_refs dans separate()"

---

### Task 5: CLI (`train-tse` + `--separation-backend`)

**Files:**
- Modify: `src/conversation_deconvolution/cli.py`
- Modify: `Makefile`

**Interfaces:**
- Produits : commande `deconvolute train-tse`, option `--separation-backend tse` sur `run` et `benchmark`.

- [ ] **Step 1: Add `train-tse` command to cli.py**

Ajouter dans `cli.py` après la commande `train` :

```python
@app.command()
def train_tse(
    epochs: int = typer.Option(30, "--epochs"),
    out: Path = typer.Option("models/tse/model.pt", "--out"),
    config_path: Path = typer.Option(None, "--config", "-c"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    cfg.tse.epochs = epochs
    from conversation_deconvolution.tse.dataset import TseDataset
    from conversation_deconvolution.tse.train import train_tse_model
    from conversation_deconvolution.synthetic.tts import PiperTts

    tts = PiperTts()
    dirs = sorted(Path("data/synthetic").glob("train_3000_*"))
    if not dirs:
        console.print("[red]✗[/red] aucun dataset train_3000_* trouvé — générez-en avec `deconvolute synth`")
        return
    dataset = TseDataset(tts, cfg.tse, dirs)
    model_path = train_tse_model(dataset, cfg.tse, str(out))
    console.print(f"[green]✓[/green] modèle TSE → {model_path}")
```

Ajouter `--separation-backend` à la commande `run` :

```python
    separation_backend: str = typer.Option(None, "--separation-backend"),
```

Et dans le corps de `run`, après le chargement de la config :

```python
    if separation_backend is not None:
        cfg.separation.backend = separation_backend
```

Ajouter `--separation-backend` à la commande `benchmark` :

```python
    separation_backend: str = typer.Option(None, "--separation-backend"),
```

Et dans le corps de `benchmark` :

```python
    if separation_backend is not None:
        cfg.separation.backend = separation_backend
```

- [ ] **Step 2: Add `train_tse_model` to tse/train.py**

Créer `src/conversation_deconvolution/tse/train.py` :

```python
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.tse.model import TseModel


def train_tse_model(dataset, config: TseConfig, out_path: str):
    model = TseModel(
        n_fft=config.n_fft, hop=config.hop, channels=config.channels,
        embed_dim=config.embed_dim, n_blocks=config.n_blocks,
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
            mix = torch.from_numpy(np.asarray(mix, dtype=np.float32)).to(device)
            target = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device)
            ref_emb = torch.from_numpy(np.asarray(ref_emb, dtype=np.float64)).to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(mix, target, ref_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(loader))
        console.print(f"epoch {epoch + 1}/{config.epochs} loss={avg:.4f}")

    torch.save(model.state_dict(), out_path)
    _save_hparams(config, out_path)
    return out_path


def _save_hparams(config: TseConfig, model_path: str):
    import yaml
    hparams_path = model_path.replace(".pt", ".yaml")
    with open(hparams_path, "w") as f:
        yaml.dump({
            "n_fft": config.n_fft, "hop": config.hop, "window": config.window,
            "n_blocks": config.n_blocks, "channels": config.channels,
            "embed_dim": config.embed_dim, "lr": config.lr,
        }, f, default_flow_style=False)
```

Note : il faut importer `console` de cli.py ou passer un logger. Pour éviter la circularité, on peut utiliser `print` ou référencer le module. Utilisons `print` pour le moment dans train.py et laissons cli.py afficher le message final.

En fait, pour rester propre, créons `tse/train.py` sans dépendance à `console` :

```python
    ...
    for epoch in range(config.epochs):
        ...
        avg = total_loss / max(1, len(loader))
        print(f"epoch {epoch + 1}/{config.epochs} loss={avg:.4f}")
    ...
```

- [ ] **Step 3: Add train-tse to Makefile**

```makefile
train-tse:
	uv run deconvolute train-tse --epochs 30 --out models/tse/model.pt

.PHONY: lint format test test-slow benchmark train train-tse
```

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest -q && make lint && uv run deconvolute train-tse --help`
Expected: PASS.

Commit : `git add src/conversation_deconvolution/cli.py src/conversation_deconvolution/tse/train.py Makefile`
`git commit -m "feat(cli): train-tse + --separation-backend"`

---

### Task 6: Tests complets

**Files:**
- Tous les tests créés dans les Tasks 1-5
- Modifier : `tests/unit/test_pipeline_separation.py` (étendre pour speaker_refs)

- [ ] **Step 1: Extend test_pipeline_separation.py**

Ajouter à `tests/unit/test_pipeline_separation.py` :

```python
def test_tse_backend_builds():
    from conversation_deconvolution.tse.model import TseModel
    from conversation_deconvolution.separation.tse_separator import TseSeparator
    from conversation_deconvolution.core.config import PipelineConfig, TseConfig
    import torch

    cfg = PipelineConfig()
    cfg.separation.backend = "tse"
    cfg.separation.enabled = True
    cfg.tse = TseConfig()
    tse_cfg = cfg.tse
    model = TseModel(n_fft=tse_cfg.n_fft, hop=tse_cfg.hop, channels=tse_cfg.channels, embed_dim=tse_cfg.embed_dim, n_blocks=tse_cfg.n_blocks)
    model.load_state_dict(torch.load(tse_cfg.model_path, map_location="cpu"), strict=False)
    model.eval()
    sep = TseSeparator(tse_cfg, model)
    assert sep is not None
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest -q -v`
Expected: ALL PASS (111 + nouveaux tests).

- [ ] **Step 3: Run lint**

Run: `make lint`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test(tse): tests complets modèle, dataset, séparateur, pipeline"
```

---

### Task 7: ADR-0011 + ROADMAP + README

**Files:**
- Create: `docs/adr/0011-tse.md`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`

- [ ] **Step 1: Write ADR-0011**

Créer `docs/adr/0011-tse.md` :

```markdown
# ADR-0011: Target Speaker Extraction (TSE) — Séparation N speakers

## Status

Accepted

## Context

Après ADR-0010, le plafond architecture est atteint sur données synthétiques :
- DER 0,091, pairwise-F1 0,651, ARI 0,480 (4 speakers, seeds 1-4)
- SepFormer limité à 2 stems → dégrade WER overlap (0,807 vs 0,768 OFF) sur 4 locuteurs réels
- La séparation conditionnelle OFF reste le comportement par défaut (ADR-0008 renforcé)

Le TSE est un modèle STFT-domain FiLM (~3M params) conditionné sur embeddings ECAPA 192-d,
entraîné sur les datasets synthétiques Piper existants (base 3000). Il produit N stems
sans limite (limite : nombre de voix Piper disponibles = 6).

## Decision

Implémenter un séparateur TSE léger en interne, entraîné sur données synthétiques déterministes,
derrière le backend `separation.backend: tse`. Le comportement par défaut reste OFF
(`separation.enabled: false`). Le TSE est activable via `--separation-backend tse`.

## Alternatives considered

- SepFormer avec fine-tuning : nécessite >2 stems, non disponible publiquement.
- Modèle externe (OpenUnmix, Demucs) : dépendance lourde, pas local-first.
- GNN sur features audio : coût élevé, peu de signaux discriminants en amont.

## Consequences

### Positive
- Séparation N speakers sans limite théorique (limite : voix Piper).
- ~3M params, entraînement local, pas de dépendance externe.
- Compatible avec le pipeline existant via `--separation-backend tse`.

### Negative
- Deux chemins de pipeline à maintenir (OFF par défaut).
- Entraînement nécessaire (~GPU) avant utilisation.
- Critère d'acceptation : WER overlap(TSE) < WER overlap(OFF).

## Reconsideration criteria
Activer par défaut si WER overlap(TSE) < WER overlap(OFF) sur le benchmark 4 speakers.
```

- [ ] **Step 2: Update ROADMAP**

Dans `docs/ROADMAP.md`, ajouter après la section M6 :

```markdown
## M6b — Target Speaker Extraction *(Phase 6, ADR-0011)*

**Livrables**
- [x] Modèle STFT-FiLM (~3M params, conditionné ECAPA 192-d)
- [x] Dataset en ligne sur `train_3000_*` existants
- [x] Entraînement `deconvolute train-tse` + checkpoints `models/tse/`
- [x] Backend `tse` dans `build_pipeline` + `--separation-backend`
- [ ] **Critère d'acceptation :** WER overlap(TSE) < WER overlap(OFF) (0,807)

**Acceptation :** non validée — approche derrière flag, OFF par défaut (ADR-0008/0011 renforcés).
```

- [ ] **Step 3: Update README**

Dans `README.md`, ajouter dans la section Statut :

```markdown
**M6b — TSE :** modèle STFT-FiLM conditionné ECAPA (~3M params), entraînement
`deconvolute train-tse`, backend activable `--separation-backend tse`.
Séparation N speakers ; OFF maintenu par défaut (ADR-0008/0011).
```

Et dans le Quickstart, ajouter :

```bash
# entraîner le modèle TSE (~2-3h GPU, 8 datasets seedés)
uv run deconvolute train-tse --epochs 30 --out models/tse/model.pt

# pipeline avec séparation TSE
deconvolute run meeting.wav -o out.json --separate --separation-backend tse
```

- [ ] **Step 4: Run make lint test and commit**

Run: `make lint test`
Expected: PASS.

Commit : `git add docs/adr/0011-tse.md docs/ROADMAP.md README.md`
`git commit -m "docs: ADR-0011 TSE + roadmap + quickstart"`

---

### Task 8: Entraînement réel + benchmark + validation

**Files:**
- Generated: `models/tse/model.pt`, `models/tse/hparams.yaml`
- Generated: `reports/benchmark_tse.md`

- [ ] **Step 1: Train TSE model**

```bash
uv run deconvolute train-tse --epochs 30 --out models/tse/model.pt
```

Attendu : loss décroissant, checkpoints sauvés. Vérifier SI-SDR sur validation set.

- [ ] **Step 2: Benchmark TSE vs OFF**

```bash
uv run deconvolute benchmark --datasets 4 --seed 1234 --separate --separation-backend tse --out reports/benchmark_tse.md
```

Critère : WER overlap(TSE) < 0,807 (baseline OFF). Si non atteint, ajuster hyperparams ou conclure.

- [ ] **Step 3: Final commit**

```bash
git add models/ reports/
git commit -m "feat(tse): modèle entraîné + benchmark TSE"
```

---

## Self-Review

- Couverture spec : §1 architecture modèle ✓ (Task 2) · §2 dataset ✓ (Task 3) · §3 entraînement ✓ (Task 5) · §4 intégration pipeline ✓ (Task 4) · §5 CLI ✓ (Task 5) · §6 tests ✓ (Task 6) · §7 ADR/ROADMAP/README ✓ (Task 7) · §8 entraînement réel + benchmark ✓ (Task 8)
- Placeholders : aucun TBD ; chaque tâche contient le code.
- Cohérence types : `TseConfig` fields identiques Tasks 1/2/3/5 · `separate(mix, regions, speaker_refs=None)` identique Tasks 4/5 · `build_pipeline` factory cohérent Tasks 4/5.
```
