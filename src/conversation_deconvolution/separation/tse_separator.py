import numpy as np
import torch

from conversation_deconvolution.core.config import TseConfig
from conversation_deconvolution.core.types import Segment, SeparatedRegion, SeparationResult
from conversation_deconvolution.tse.model import TseModel


class TseSeparator:
    def __init__(self, config: TseConfig, model: TseModel):
        self.cfg = config
        self.model = model
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def separate(
        self, mix: np.ndarray, regions: list[Segment], speaker_refs: dict | None = None
    ) -> SeparationResult:
        mix_arr = np.asarray(mix, dtype=np.float32)
        sr = 16000
        if not speaker_refs:
            return SeparationResult(
                mix=mix_arr.copy(),
                regions=[SeparatedRegion(segment=r, stems=[]) for r in regions],
            )
        keys = sorted(speaker_refs.keys())
        embs = []
        for k in keys:
            emb = np.asarray(speaker_refs[k], dtype=np.float64)
            norm = float(np.linalg.norm(emb)) or 1.0
            embs.append(torch.from_numpy(emb / norm).float().to(self.device))
        ref_embs = torch.stack(embs)
        with torch.no_grad():
            mix_tensor = torch.from_numpy(mix_arr).to(self.device).unsqueeze(0)
            all_stems: list[list[np.ndarray]] = [[] for _ in regions]
            for i, region in enumerate(regions):
                s = max(0, int(region.start * sr))
                e = min(len(mix_arr), int(region.end * sr))
                seg = mix_tensor[:, s:e]
                if seg.shape[-1] < self.cfg.n_fft:
                    continue
                for emb in ref_embs:
                    mask = self.model(seg, emb.unsqueeze(0))
                    est = self.model.apply_mask(seg, mask)
                    stem = est.squeeze(0).cpu().numpy().astype(np.float32)
                    all_stems[i].append(stem)
        regions_out = [SeparatedRegion(segment=r, stems=s) for r, s in zip(regions, all_stems)]
        return SeparationResult(
            mix=mix_arr.copy(), regions=regions_out, meta={"num_speakers": len(keys)}
        )
