import numpy as np
import torch

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.core.types import Segment
from conversation_deconvolution.separation.tse_separator import TseSeparator
from conversation_deconvolution.tse.model import TseModel


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
    assert result.meta["num_speakers"] == 1
    assert result.regions[0].stems[0].shape[0] == 16000


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
    refs = {
        "A": np.random.randn(192).astype(np.float64),
        "B": np.random.randn(192).astype(np.float64),
    }
    result = sep.separate(mix, [_fake_region()], refs)
    assert len(result.regions[0].stems) == 2
    assert result.meta["num_speakers"] == 2
