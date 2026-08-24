import numpy as np

from conversation_deconvolution.core.types import Segment, SeparatedRegion, SeparationResult


class SepformerSeparator:
    def __init__(self, config=None):
        self.cfg = config
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from speechbrain.inference.separation import SepformerSeparation

            device = (self.cfg.device if self.cfg and self.cfg.device else None) or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            name = self.cfg.model_name if self.cfg else "speechbrain/sepformer-whamr16k"
            savedir = f"models/{name.split('/')[-1]}"
            self._model = SepformerSeparation.from_hparams(
                source=name, savedir=savedir, run_opts={"device": device}
            )
        return self._model

    def separate(self, mix: np.ndarray, regions: list[Segment]) -> SeparationResult:
        import torch

        min_sec = self.cfg.min_region_sec if self.cfg else 0.4
        pad = self.cfg.pad_sec if self.cfg else 0.25
        mix = np.asarray(mix, dtype=np.float32)
        separated: list[SeparatedRegion] = []
        for seg in regions:
            if seg.duration < min_sec:
                continue
            s = max(0, int((seg.start - pad) * 16000))
            e = min(len(mix), int((seg.end + pad) * 16000))
            chunk = mix[s:e]
            model = self._load()
            with torch.no_grad():
                est = model.separate_batch(torch.tensor(chunk, dtype=torch.float32)[None, :])
            stems = [
                np.asarray(est[0, :, i].cpu(), dtype=np.float32)
                for i in range(est.shape[-1])
            ]
            stems = [self._fit(stem, len(chunk)) for stem in stems]
            separated.append(
                SeparatedRegion(segment=Segment(s / 16000, e / 16000), stems=stems)
            )
        return SeparationResult(mix=mix.copy(), regions=separated)

    @staticmethod
    def _fit(signal: np.ndarray, n: int) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if len(signal) >= n:
            return signal[:n]
        return np.pad(signal, (0, n - len(signal)))
