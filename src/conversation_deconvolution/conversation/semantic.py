import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float64)
