import numpy as np

from conversation_deconvolution.core.types import Segment


class Separator:
    def separate(self, mix: np.ndarray, regions: list[Segment]) -> list[np.ndarray]:
        raise NotImplementedError


class PassthroughSeparator(Separator):
    def separate(self, mix: np.ndarray, regions: list[Segment]) -> list[np.ndarray]:
        return [np.asarray(mix, dtype=np.float32).copy()]
