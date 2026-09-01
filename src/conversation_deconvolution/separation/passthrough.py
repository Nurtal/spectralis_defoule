import numpy as np

from conversation_deconvolution.core.types import Segment, SeparationResult


class Separator:
    def separate(
        self, mix: np.ndarray, regions: list[Segment], speaker_refs: dict | None = None
    ) -> SeparationResult:
        raise NotImplementedError


class PassthroughSeparator(Separator):
    def separate(
        self, mix: np.ndarray, regions: list[Segment], speaker_refs: dict | None = None
    ) -> SeparationResult:
        return SeparationResult(mix=np.asarray(mix, dtype=np.float32).copy())
