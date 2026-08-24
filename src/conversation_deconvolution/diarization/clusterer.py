import numpy as np
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeClusterer:
    def __init__(self, distance_threshold: float = 0.75):
        self.distance_threshold = distance_threshold

    def fit_predict(self, embeddings: np.ndarray, n_speakers: int | None = None) -> np.ndarray:
        X = np.asarray(embeddings, dtype=np.float64)
        if len(X) == 1:
            return np.zeros(1, dtype=int)
        if n_speakers is not None:
            model = AgglomerativeClustering(
                n_clusters=n_speakers, metric="cosine", linkage="average"
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average",
            )
        return model.fit_predict(X)
