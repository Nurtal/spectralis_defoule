import numpy as np

from conversation_deconvolution.core.types import Segment
from conversation_deconvolution.separation.passthrough import (
    PassthroughSeparator,
    SeparatedRegion,
    SeparationResult,
)


def test_passthrough_mix_only_and_isolated():
    mix = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    sep = PassthroughSeparator()
    out = sep.separate(mix, [Segment(0, 0.5)])
    assert isinstance(out, SeparationResult)
    assert out.regions == []
    np.testing.assert_array_equal(out.mix, mix)
    out.mix[0] = 99.0
    assert mix[0] == 0.1


def test_separated_region_holds_stems():
    stems = [np.zeros(4, dtype=np.float32)]
    region = SeparatedRegion(segment=Segment(1.0, 2.0), stems=stems)
    assert region.segment.duration == 1.0
    assert len(region.stems) == 1
