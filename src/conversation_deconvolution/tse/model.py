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


class TseModel(nn.Module):
    def __init__(
        self, n_fft=512, hop=256, window="hann", n_blocks=3, channels=128, embed_dim=192,
        freq_bands=32,
    ):
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
        self.cond_conv = nn.Conv2d(channels + embed_dim, channels, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.gamma_fc = nn.Linear(embed_dim, channels)
        self.beta_fc = nn.Linear(embed_dim, channels)
        self.sigmoid = nn.Sigmoid()
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, mix, ref_emb):
        spec = _stft(mix, self.n_fft, self.hop, self.window)
        real = spec.real.unsqueeze(1)
        imag = spec.imag.unsqueeze(1)
        x = torch.cat([real, imag], dim=1)
        x = self.encoder(x)
        gamma = self.gamma_fc(ref_emb).view(-1, self.channels, 1, 1)
        beta = self.beta_fc(ref_emb).view(-1, self.channels, 1, 1)
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
        return -_si_sdr(est, target).mean()


def _si_sdr(estimate, reference):
    eps = 1e-8
    ref_energy = torch.sum(reference**2, dim=-1, keepdim=True) + eps
    proj = torch.sum(reference * estimate, dim=-1, keepdim=True)
    alpha = proj / ref_energy
    residual = estimate - alpha * reference
    return 10 * torch.log10(ref_energy / (torch.sum(residual**2, dim=-1, keepdim=True) + eps))
