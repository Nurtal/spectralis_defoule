import typing

import numpy as np

from conversation_deconvolution.core.config import PipelineConfig
from conversation_deconvolution.core.types import (
    Segment,
    SeparatedRegion,
    SeparationResult,
    SpeakerTurn,
    VadResult,
)

SR = 16000


def _tone(freq, dur, amp):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeVad:
    def detect(self, audio):
        return VadResult(
            segments=[Segment(0.0, 3.0)],
            frame_probs=np.zeros(150),
            frame_rate=50.0,
        )


class FakeDiarizer:
    def __init__(self):
        self.vad = FakeVad()

    def diarize(self, audio):
        return [
            SpeakerTurn("SPEAKER_00", 0.0, 2.0),
            SpeakerTurn("SPEAKER_01", 1.0, 3.0),
        ], []


class LoudSeparator:
    def separate(self, mix, regions):
        assert regions and regions[0] == Segment(1.0, 2.0)
        stems = [_tone(440, 1.0, 0.9), _tone(880, 1.0, 0.3)]
        return SeparationResult(
            mix=np.asarray(mix).copy(),
            regions=[SeparatedRegion(segment=Segment(1.0, 2.0), stems=stems)],
        )


class BandEmbedder:
    """Embeds a 1-D signal with power stats so amplitudes separate."""

    dim = 8

    def encode(self, signals):
        out = []
        for x in signals:
            x = np.asarray(x, dtype=np.float64)
            v = np.zeros(self.dim)
            v[0] = float(np.mean(x * x))
            v[1] = float(np.mean(x**4))
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return np.array(out)


class RecordingAsr:
    def __init__(self):
        self.inputs = []

    def transcribe(self, segment, language=None):
        import types

        self.inputs.append(np.asarray(segment))
        rms_mid = float(np.sqrt(np.mean(segment[SR : 2 * SR] ** 2)))
        dominant = "low" if rms_mid > 0.5 else "high"
        return types.SimpleNamespace(text=dominant, confidence=0.9, language="fr")


def make_result(asr, separator):
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    pipeline = DeconvolutionPipeline(
        diarizer=FakeDiarizer(),
        separator=separator,
        asr=asr,
        reconstructor=HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig()),
        config=_cfg(),
        stem_embedder=BandEmbedder(),
    )
    return pipeline.run(np.zeros(3 * SR, dtype=np.float32))


def _cfg():
    cfg = PipelineConfig()
    cfg.asr.context_pad_sec = 0.0
    return cfg


class _NullEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 4))


def test_overlap_turns_get_assigned_stems():
    class MixSeparator(LoudSeparator):
        def separate(self, mix_, regions):
            out = super().separate(mix_, regions)
            out.mix = np.asarray(mix_).copy()
            return out

    asr = RecordingAsr()
    result = make_result(asr, MixSeparator())
    assert len(result.utterances) == 2
    assert result.utterances[0].text == "low"
    assert result.utterances[1].text == "high"


def test_no_embedder_falls_back_to_mix():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    mix = np.concatenate([_tone(440, 1.0, 0.9), np.zeros(SR, np.float32), _tone(880, 1.0, 0.3)])
    asr = RecordingAsr()
    pipeline = DeconvolutionPipeline(
        diarizer=FakeDiarizer(),
        separator=LoudSeparator(),
        asr=asr,
        reconstructor=HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig()),
        config=_cfg(),
        stem_embedder=None,
    )
    result = pipeline.run(mix)
    assert len(result.utterances) == 2
    assert all(u.text == "high" for u in result.utterances)


def test_diarizer_reported_overlaps_reach_separator():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    class AwareDiarizer(FakeDiarizer):
        overlap_regions_: typing.ClassVar = [Segment(1.0, 2.0)]

    class CapturingSeparator(LoudSeparator):
        def __init__(self):
            self.seen = None

        def separate(self, mix_, regions):
            self.seen = list(regions)
            return super().separate(mix_, regions)

    sep = CapturingSeparator()
    asr = RecordingAsr()
    pipeline = DeconvolutionPipeline(
        diarizer=AwareDiarizer(),
        separator=sep,
        asr=asr,
        reconstructor=HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig()),
        config=_cfg(),
        stem_embedder=BandEmbedder(),
    )
    result = pipeline.run(np.zeros(3 * SR, dtype=np.float32))
    assert sep.seen == [Segment(1.0, 2.0)]
    assert result.overlaps == [Segment(1.0, 2.0)]
