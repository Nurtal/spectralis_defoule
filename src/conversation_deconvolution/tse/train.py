from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.tse.model import TseModel


def train_tse_model(dataset, config: TseConfig, out_path: str) -> str:
    model = TseModel(
        n_fft=config.n_fft,
        hop=config.hop,
        window=config.window,
        channels=config.channels,
        embed_dim=config.embed_dim,
        n_blocks=config.n_blocks,
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
            loss = -model.compute_loss(mix, target, ref_emb).mean()
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
                "lr": config.lr,
            },
            f,
            default_flow_style=False,
        )
