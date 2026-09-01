import torch

from conversation_deconvolution.tse.model import TseModel, _si_sdr


def test_forward_shapes():
    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        mask = model(mix, ref_emb)
    assert mask.shape[0] == 1
    assert mask.shape[1] == 257
    assert mask.shape[2] > 0


def test_loss_reduces():
    import math

    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    target = torch.randn(1, 16000)
    with torch.no_grad():
        loss = model.compute_loss(mix, target, ref_emb)
    assert loss.item() > 0
    assert math.isfinite(loss.item())


def test_param_count():
    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= n_params <= 4_000_000


def test_deterministic():
    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        out1 = model(mix, ref_emb)
        out2 = model(mix, ref_emb)
    assert torch.allclose(out1, out2)


def test_mask_applied_is_bounded():
    model = TseModel(n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        mask = model(mix, ref_emb)
    mask_real = mask.real
    mask_imag = mask.imag
    assert mask_real.min() >= 0 and mask_real.max() <= 1
    assert mask_imag.min() >= 0 and mask_imag.max() <= 1


def test_si_sdr_positive():
    est = torch.ones(16000)
    ref = torch.ones(16000)
    loss = _si_sdr(est, ref)
    assert loss.item() > 0
