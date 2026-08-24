import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples


class AgglomerativeClusterer:
    def __init__(self, distance_threshold: float = 0.75, max_speakers: int = 8):
        self.distance_threshold = distance_threshold
        self.max_speakers = max_speakers

    def fit_predict(self, embeddings: np.ndarray, n_speakers: int | None = None) -> np.ndarray:
        X = np.asarray(embeddings, dtype=np.float64)
        if len(X) == 1:
            return np.zeros(1, dtype=int)
        if n_speakers is not None:
            model = AgglomerativeClustering(
                n_clusters=n_speakers, metric="cosine", linkage="average"
            )
            return model.fit_predict(X)
        return self._auto_k(X)

    def _fixed(self, X: np.ndarray, k: int) -> np.ndarray:
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        return model.fit_predict(X)

    def _auto_k(self, X: np.ndarray) -> np.ndarray:
        if len(X) <= self.max_speakers:
            labels = self._threshold(X)
            return labels if len(set(labels)) > 1 else np.zeros(len(X), dtype=int)
        best_labels = None
        best_score = -np.inf
        for k in range(2, self.max_speakers + 1):
            labels = self._fixed(X, k)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_samples(X, labels, metric="cosine").mean())
            if score > best_score:
                best_score = score
                best_labels = labels
        if best_labels is None:
            return self._threshold(X)
        return best_labels

    def _threshold(self, X: np.ndarray) -> np.ndarray:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(X)
