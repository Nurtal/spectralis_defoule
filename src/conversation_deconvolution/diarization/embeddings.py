import numpy as np


class EcapaEmbedder:
    def __init__(self, device: str = "cuda"):
        self.device = device if _torch_cuda_available(device) else "cpu"
        self._encoder = None

    def _ensure_encoder(self):
        if self._encoder is None:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            self._encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain-ecapa",
                run_opts={"device": self.device},
            )
        return self._encoder

    def embed(self, segment: np.ndarray) -> np.ndarray:
        import torch

        encoder = self._ensure_encoder()
        wav = torch.from_numpy(np.asarray(segment, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = encoder.encode_batch(wav)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm


def _torch_cuda_available(device: str) -> bool:
    if device != "cuda":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
