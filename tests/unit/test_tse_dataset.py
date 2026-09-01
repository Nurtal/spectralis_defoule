import json

import numpy as np
import pytest

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.tse.dataset import TseDataset, _bandlimit_noise


class FakeTts:
    def synthesize(self, text, voice):
        n = int(0.5 * 16000)
        t = np.arange(n) / 16000
        freq = 200 + abs(hash(voice)) % 300
        audio = 0.4 * np.sin(2 * np.pi * freq * t)
        return audio.astype(np.float32), 16000


def _fake_embedding(audio):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(192)
    v = v / (float(np.linalg.norm(v)) or 1.0)
    return v.astype(np.float64)


def _make_gt_dir(tmp_path, name="ds", speakers=None):
    if speakers is None:
        speakers = ["spk_A", "spk_B", "spk_C", "spk_D"]
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    utterances = []
    uid = 0
    for rep in range(2):
        for spk in speakers:
            utterances.append(
                {
                    "id": f"u_{uid}",
                    "speaker": spk,
                    "start": 0.0,
                    "end": 0.4,
                    "text": f"hello world {uid} for {spk}",
                }
            )
            uid += 1
    gt = {"conversations": [{"id": "c0", "participants": speakers, "utterances": utterances}]}
    (d / "ground_truth.json").write_text(json.dumps(gt))
    return d


def test_batch_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(TseDataset, "_compute_embedding", lambda self, a: _fake_embedding(a))
    d = _make_gt_dir(tmp_path, "ds_shapes")
    cfg = TseConfig(batch_size=2, snr_low=20.0, snr_high=20.0, noise_bandwidth=0.0)
    ds = TseDataset(FakeTts(), cfg, [str(d)])
    assert len(ds) > 0
    mix, target, ref_emb, n_spk = ds[0]
    assert mix.shape == (16000,)
    assert target.shape == (16000,)
    assert mix.dtype == np.float32
    assert target.dtype == np.float32
    assert ref_emb.shape == (192,)
    assert 2 <= n_spk <= 4
    assert np.all(np.isfinite(mix))
    assert np.all(np.isfinite(target))
    assert np.all(np.isfinite(ref_emb))
    assert not np.any(np.isnan(mix))
    assert not np.any(np.isnan(ref_emb))


def test_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(TseDataset, "_compute_embedding", lambda self, a: _fake_embedding(a))
    d = _make_gt_dir(tmp_path, "ds_det")
    cfg = TseConfig(batch_size=2, snr_low=20.0, snr_high=20.0, noise_bandwidth=0.0)
    ds = TseDataset(FakeTts(), cfg, [str(d)])
    a_mix, a_tgt, a_emb, a_n = ds[3]
    b_mix, b_tgt, b_emb, b_n = ds[3]
    assert np.allclose(a_mix, b_mix)
    assert np.allclose(a_tgt, b_tgt)
    assert np.allclose(a_emb, b_emb)
    assert a_n == b_n
    c_mix, _, _, _ = ds[4]
    assert not np.allclose(a_mix, c_mix)


def test_no_nan_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(TseDataset, "_compute_embedding", lambda self, a: _fake_embedding(a))
    d = _make_gt_dir(tmp_path, "ds_nan")
    cfg = TseConfig(batch_size=4, snr_low=10.0, snr_high=10.0, noise_bandwidth=3400.0)
    ds = TseDataset(FakeTts(), cfg, [str(d)])
    mix, target, ref_emb, _ = ds[0]
    assert not np.any(np.isnan(mix))
    assert not np.any(np.isnan(target))
    assert not np.any(np.isnan(ref_emb))
    assert np.max(np.abs(mix)) < 10.0
    assert np.max(np.abs(target)) < 10.0
    assert float(np.linalg.norm(ref_emb)) == pytest.approx(1.0, rel=1e-5)


def test_len_and_empty_dirs(tmp_path):
    cfg = TseConfig(batch_size=2)
    with pytest.raises(ValueError):
        TseDataset(FakeTts(), cfg, [])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        TseDataset(FakeTts(), cfg, [str(empty)])
    d = _make_gt_dir(tmp_path, "ds_len")
    ds = TseDataset(FakeTts(), cfg, [str(d)])
    expected = max(1, 8 // cfg.batch_size)
    assert len(ds) == expected
    d2 = _make_gt_dir(tmp_path, "ds_len2")
    ds2 = TseDataset(FakeTts(), cfg, [str(d), str(d2)])
    assert len(ds2) == max(1, 16 // cfg.batch_size)


def test_bandlimit_noise():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(16000).astype(np.float32)
    out_zero = _bandlimit_noise(noise, 0.0, sr=16000)
    assert out_zero.shape == noise.shape
    assert np.allclose(out_zero, noise)
    out_neg = _bandlimit_noise(noise, -100.0, sr=16000)
    assert np.allclose(out_neg, noise)
    out = _bandlimit_noise(noise, 3400.0, sr=16000)
    assert out.shape == noise.shape
    assert np.all(np.isfinite(out))
    assert not np.allclose(out, noise)


def test_from_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(TseDataset, "_compute_embedding", lambda self, a: _fake_embedding(a))
    d = _make_gt_dir(tmp_path, "ds_existing")
    cfg = TseConfig(batch_size=2, snr_low=20.0, snr_high=20.0, noise_bandwidth=0.0)
    ds = TseDataset.from_existing(FakeTts(), cfg, str(d))
    assert len(ds) > 0
    mix, target, ref_emb, n_spk = ds[0]
    assert mix.shape == (16000,)
    assert target.shape == (16000,)
    assert ref_emb.shape == (192,)
    assert 2 <= n_spk <= 4


def test_ref_exclusive_and_snr_range(tmp_path, monkeypatch):
    monkeypatch.setattr(TseDataset, "_compute_embedding", lambda self, a: _fake_embedding(a))
    d = _make_gt_dir(tmp_path, "ds_excl")
    cfg = TseConfig(batch_size=2, snr_low=10.0, snr_high=20.0, noise_bandwidth=0.0)
    ds = TseDataset(FakeTts(), cfg, [str(d)])
    mix1, _, _, _ = ds[1]
    mix2, _, _, _ = ds[1]
    assert np.allclose(mix1, mix2)
    for idx in range(5):
        mix, target, _, n_spk = ds[idx]
        assert 2 <= n_spk <= 4
        assert np.any(target != 0)
        assert np.all(np.isfinite(mix))
