import numpy as np

from conversation_deconvolution.core.types import Segment
from conversation_deconvolution.separation.passthrough import PassthroughSeparator


def test_passthrough_identity_and_isolation():
    mix = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    sep = PassthroughSeparator()
    out = sep.separate(mix, [Segment(0, 0.5)])
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], mix)
    out[0][0] = 99.0
    assert mix[0] == 0.1
