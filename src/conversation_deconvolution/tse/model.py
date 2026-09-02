import torch
import torch.nn.functional as F
from torch import nn


def _stft(mix, n_fft=512, hop=256, window="hann"):
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    win = torch.hann_window(n_fft, device=mix.device)
    spec = torch.stft(mix, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    return spec


def _istft(masked_spec, mix, n_fft=512, hop=256, window="hann"):
    win = torch.hann_window(n_fft, device=masked_spec.device)
    return torch.istft(
        masked_spec, n_fft=n_fft, hop_length=hop, window=win, length=mix.shape[-1]
    )


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
            gamma = F.interpolate(
                gamma, size=(x.shape[2], 1), mode="bilinear", align_corners=False
            )
            beta = F.interpolate(
                beta, size=(x.shape[2], 1), mode="bilinear", align_corners=False
            )
        out = F.relu(self.bn1(self.conv1(x)))
        out = gamma * out + beta
        out = self.bn2(self.conv2(out))
        out = gamma * out + beta
        return F.relu(out + residual)


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


class TseModel(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop=256,
        window="hann",
        n_blocks=3,
        channels=128,
        embed_dim=192,
        freq_bands=32,
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


def _si_sdr(estimate, reference):
    eps = 1e-8
    ref_energy = torch.sum(reference**2, dim=-1, keepdim=True) + eps
    proj = torch.sum(reference * estimate, dim=-1, keepdim=True)
    alpha = proj / ref_energy
    residual = estimate - alpha * reference
    return 10 * torch.log10(ref_energy / (torch.sum(residual**2, dim=-1, keepdim=True) + eps))
