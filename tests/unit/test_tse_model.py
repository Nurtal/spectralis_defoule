import torch

from conversation_deconvolution.tse.model import TseModel, _si_sdr


def test_forward_shapes():
    model = TseModel(
        n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32
    )
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

    model = TseModel(
        n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32
    )
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    target = torch.randn(1, 16000)
    with torch.no_grad():
        loss = model.compute_loss(mix, target, ref_emb)
    assert math.isfinite(loss.item())


def test_param_count():
    model = TseModel(
        n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= n_params <= 2_500_000


def test_deterministic():
    model = TseModel(
        n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32
    )
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        out1 = model(mix, ref_emb)
        out2 = model(mix, ref_emb)
    assert torch.allclose(out1, out2)


def test_mask_applied_is_bounded():
    model = TseModel(
        n_fft=512, hop=256, channels=128, embed_dim=192, n_blocks=3, freq_bands=32
    )
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    with torch.no_grad():
        mask = model(mix, ref_emb)
    if mask.is_complex():
        mask_real = mask.real
        mask_imag = mask.imag
        assert mask_real.min() >= 0 and mask_real.max() <= 1
        assert mask_imag.min() >= 0 and mask_imag.max() <= 1
    else:
        assert mask.min() >= 0 and mask.max() <= 1


def test_si_sdr_positive():
    est = torch.ones(16000)
    ref = torch.ones(16000)
    loss = _si_sdr(est, ref)
    assert loss.item() > 0


def test_film_block_v2_per_freq_shapes():
    from conversation_deconvolution.tse.model import FilmBlockV2

    block = FilmBlockV2(channels=64)
    x = torch.randn(2, 64, 257, 50)
    gamma = torch.randn(2, 64, 32, 1)
    beta = torch.randn(2, 64, 32, 1)
    out = block(x, gamma, beta)
    assert out.shape == x.shape


def test_conditioning_mlp_output_shape():
    from conversation_deconvolution.tse.model import ConditioningMLP

    mlp = ConditioningMLP(embed_dim=192, channels=64, freq_bands=32)
    emb = torch.randn(2, 192)
    gamma, beta = mlp(emb)
    assert gamma.shape == (2, 64, 32, 1)
    assert beta.shape == (2, 64, 32, 1)


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


def test_param_count_default():
    from conversation_deconvolution.tse.model import TseModel

    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3, freq_bands=32)
    n_params = sum(p.numel() for p in model.parameters())
    assert 500_000 <= n_params <= 1_000_000


def test_contrastive_loss():
    import math

    from conversation_deconvolution.tse.model import TseModel

    model = TseModel(n_fft=512, hop=256, channels=64, embed_dim=192, n_blocks=3, freq_bands=32)
    model.eval()
    mix = torch.randn(1, 16000)
    ref_emb = torch.randn(1, 192)
    target = torch.randn(1, 16000)
    with torch.no_grad():
        loss = model.compute_loss(mix, target, ref_emb, lambda_rec=0.5, lambda_sim=0.5)
    assert math.isfinite(loss.item())
