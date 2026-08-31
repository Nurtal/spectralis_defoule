import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft(mix, n_fft=512, hop=256, window="hann"):
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    win = torch.hann_window(n_fft, device=mix.device)
    spec = torch.stft(mix, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    return spec


def _istft(masked_spec, mix, n_fft=512, hop=256, window="hann"):
    win = torch.hann_window(n_fft, device=masked_spec.device)
    return torch.istft(masked_spec, n_fft=n_fft, hop_length=hop, window=win, length=mix.shape[-1])


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
    def __init__(self, n_fft=512, hop=256, window="hann", n_blocks=3, channels=128, embed_dim=192):
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
        real = spec.real.unsqueeze(1)
        imag = spec.imag.unsqueeze(1)
        x = torch.cat([real, imag], dim=1)
        x = self.encoder(x)
        gamma = self.gamma_fc(ref_emb).view(-1, self.channels, 1, 1)
        beta = self.beta_fc(ref_emb).view(-1, self.channels, 1, 1)
        for block in self.blocks:
            x = block(x, gamma, beta)
        mask_parts = self.decoder(x)
        mask_real = self.sigmoid(mask_parts[:, 0])
        mask_imag = self.sigmoid(mask_parts[:, 1])
        mask = torch.complex(mask_real, mask_imag)
        return mask

    def apply_mask(self, mix, mask):
        spec = _stft(mix, self.n_fft, self.hop, self.window)
        masked_spec = mask * spec
        return _istft(masked_spec, mix, self.n_fft, self.hop, self.window)

    def compute_loss(self, mix, target, ref_emb):
        mask = self.forward(mix, ref_emb)
        est = self.apply_mask(mix, mask)
        return _si_sdr(est, target)


def _si_sdr(estimate, reference):
    eps = 1e-8
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(reference * estimate, dim=-1, keepdim=True)
    alpha = proj / ref_energy
    residual = estimate - alpha * reference
    return 10 * torch.log10(ref_energy / (torch.sum(residual ** 2, dim=-1, keepdim=True) + eps))