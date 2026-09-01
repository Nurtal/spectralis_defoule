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


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (float(np.linalg.norm(v)) or 1.0)


class FakeDiarizer:
    speaker_centroids_: typing.ClassVar = {
        0: _unit([0.405, 0.246] + [0.0] * 6),
        1: _unit([0.045, 0.003] + [0.0] * 6),
    }

    def __init__(self):
        self.vad = FakeVad()

    def diarize(self, audio):
        return [
            SpeakerTurn("SPEAKER_00", 0.0, 2.0),
            SpeakerTurn("SPEAKER_01", 1.0, 3.0),
        ], []


class FakeTseDiarizer(FakeDiarizer):
    speaker_centroids_: typing.ClassVar = {
        0: _unit(np.random.default_rng(0).standard_normal(192)),
        1: _unit(np.random.default_rng(1).standard_normal(192)),
    }


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


class FakeEcapaEmbedder:
    dim = 192

    def encode(self, signals):
        out = []
        for x in signals:
            x = np.asarray(x, dtype=np.float64)
            v = np.zeros(192)
            v[0] = float(np.mean(x * x))
            v[1] = float(np.mean(x**4)) if x.size else 0.0
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return np.array(out)


class RecordingAsr:
    def __init__(self):
        self.inputs = []

    def transcribe(self, segment, language=None):
        import types

        self.inputs.append(np.asarray(segment))
        power = float(np.mean(np.asarray(segment) ** 2))
        dominant = "low" if power > 0.1 else "high"
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
    assert len(asr.inputs) == 2
    assert len(asr.inputs[0]) == 2 * SR


def test_unassigned_turn_keeps_mix_audio():
    class QuietSeparator(LoudSeparator):
        def separate(self, mix_, regions):
            out = super().separate(mix_, regions)
            out.regions = [
                SeparatedRegion(segment=r.segment, stems=[s * 0.01 for s in r.stems])
                for r in out.regions
            ]
            return out

    asr = RecordingAsr()
    result = make_result(asr, QuietSeparator())
    assert len(result.utterances) == 2
    assert all(len(x) == 2 * SR for x in asr.inputs)


def test_no_embedder_falls_back_to_mix():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    mix = np.concatenate(
        [_tone(440, 1.0, 0.3), np.zeros(SR, np.float32), _tone(880, 1.0, 0.3)]
    )
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


def test_tse_backend_builds():
    from conversation_deconvolution.core.config import TseConfig
    from conversation_deconvolution.separation.tse_separator import TseSeparator
    from conversation_deconvolution.tse.model import TseModel

    cfg = TseConfig()
    model = TseModel(
        n_fft=cfg.n_fft,
        hop=cfg.hop,
        channels=cfg.channels,
        embed_dim=cfg.embed_dim,
        n_blocks=cfg.n_blocks,
    )
    model.eval()
    sep = TseSeparator(cfg, model)
    assert sep is not None
    assert sep.cfg is cfg
    assert sep.model is model


def test_tse_separator_via_pipeline_produces_stems():
    import torch

    from conversation_deconvolution.conversation.reconstructor import (
        HeuristicReconstructor,
    )
    from conversation_deconvolution.core.config import ReconstructionConfig, TseConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline
    from conversation_deconvolution.separation.tse_separator import TseSeparator
    from conversation_deconvolution.tse.model import TseModel

    tse_cfg = TseConfig()
    model = TseModel(
        n_fft=tse_cfg.n_fft,
        hop=tse_cfg.hop,
        channels=tse_cfg.channels,
        embed_dim=tse_cfg.embed_dim,
        n_blocks=tse_cfg.n_blocks,
    )
    model.eval()
    sep = TseSeparator(tse_cfg, model)
    diarizer = FakeTseDiarizer()
    asr = RecordingAsr()
    pipeline = DeconvolutionPipeline(
        diarizer=diarizer,
        separator=sep,
        asr=asr,
        reconstructor=HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig()),
        config=_cfg(),
        stem_embedder=FakeEcapaEmbedder(),
    )
    audio = np.zeros(3 * SR, dtype=np.float32)
    with torch.no_grad():
        result = pipeline.run(audio)
    assert len(result.utterances) == 2
    assert len(result.overlaps) == 1


def test_build_speaker_refs_str_keys():
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    diarizer = FakeDiarizer()
    pipeline = DeconvolutionPipeline(
        diarizer=diarizer,
        separator=LoudSeparator(),
        asr=RecordingAsr(),
        reconstructor=None,
        config=_cfg(),
        stem_embedder=None,
    )
    refs = pipeline._build_speaker_refs()
    assert refs is not None
    assert set(refs.keys()) == {"0", "1"}
    for k, v in refs.items():
        assert isinstance(k, str)
        assert isinstance(v, np.ndarray)


def test_build_speaker_refs_none_without_centroids():
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    class NoCentroidDiarizer(FakeDiarizer):
        speaker_centroids_ = None

    pipeline = DeconvolutionPipeline(
        diarizer=NoCentroidDiarizer(),
        separator=LoudSeparator(),
        asr=RecordingAsr(),
        reconstructor=None,
        config=_cfg(),
        stem_embedder=None,
    )
    assert pipeline._build_speaker_refs() is None

    class EmptyCentroidDiarizer(FakeDiarizer):
        speaker_centroids_: typing.ClassVar = {}

    pipeline2 = DeconvolutionPipeline(
        diarizer=EmptyCentroidDiarizer(),
        separator=LoudSeparator(),
        asr=RecordingAsr(),
        reconstructor=None,
        config=_cfg(),
        stem_embedder=None,
    )
    assert pipeline2._build_speaker_refs() is None


def test_pipeline_passes_speaker_refs_to_separator():
    from conversation_deconvolution.conversation.reconstructor import (
        HeuristicReconstructor,
    )
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    class CapturingTseSeparator:
        def __init__(self):
            self.seen_refs = None
            self.seen_regions = None

        def separate(self, mix, regions, speaker_refs=None):
            self.seen_refs = speaker_refs
            self.seen_regions = list(regions)
            return SeparationResult(
                mix=np.asarray(mix).copy(),
                regions=[SeparatedRegion(segment=r, stems=[]) for r in regions],
            )

    sep = CapturingTseSeparator()
    diarizer = FakeDiarizer()
    pipeline = DeconvolutionPipeline(
        diarizer=diarizer,
        separator=sep,
        asr=RecordingAsr(),
        reconstructor=HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig()),
        config=_cfg(),
        stem_embedder=BandEmbedder(),
    )
    pipeline.run(np.zeros(3 * SR, dtype=np.float32))
    assert sep.seen_refs is not None
    assert set(sep.seen_refs.keys()) == {"0", "1"}
    assert sep.seen_regions == [Segment(1.0, 2.0)]


def test_pipeline_fallback_for_separator_without_speaker_refs():
    from conversation_deconvolution.conversation.reconstructor import (
        HeuristicReconstructor,
    )
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.pipeline import DeconvolutionPipeline

    class OldSeparator:
        def separate(self, mix, regions):
            return SeparationResult(
                mix=np.asarray(mix).copy(),
                regions=[SeparatedRegion(segment=r, stems=[]) for r in regions],
            )

    pipeline = DeconvolutionPipeline(
        diarizer=FakeDiarizer(),
        separator=OldSeparator(),
        asr=RecordingAsr(),
        reconstructor=HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig()),
        config=_cfg(),
        stem_embedder=BandEmbedder(),
    )
    result = pipeline.run(np.zeros(3 * SR, dtype=np.float32))
    assert len(result.utterances) == 2


def test_build_pipeline_tse_backend_without_checkpoint(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    import conversation_deconvolution.pipeline as pipe_mod
    from conversation_deconvolution.core.config import PipelineConfig
    from conversation_deconvolution.separation.tse_separator import TseSeparator

    monkeypatch.setattr(
        "conversation_deconvolution.audio.vad.SileroVad",
        lambda cfg: MagicMock(),
    )
    monkeypatch.setattr(
        "conversation_deconvolution.diarization.embeddings.EcapaEmbedder",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "conversation_deconvolution.diarization.clusterer.AgglomerativeClusterer",
        lambda thr: MagicMock(),
    )
    monkeypatch.setattr(
        "conversation_deconvolution.diarization.diarizer.SpeakerDiarizer",
        lambda vad, emb, clus, cfg: FakeTseDiarizer(),
    )
    monkeypatch.setattr(
        "conversation_deconvolution.conversation.semantic.SentenceTransformerEmbedder",
        lambda model: MagicMock(encode=lambda texts: np.zeros((len(texts), 4))),
    )
    monkeypatch.setattr(
        "conversation_deconvolution.asr.faster_whisper_asr.FasterWhisperAsr",
        lambda cfg: MagicMock(
            transcribe=lambda seg, language=None: MagicMock(
                text="hi", confidence=0.9, language="fr"
            )
        ),
    )
    monkeypatch.setattr(
        "conversation_deconvolution.pipeline.build_reconstructor",
        lambda cfg, emb: MagicMock(reconstruct=lambda utterances: []),
    )

    cfg = PipelineConfig()
    cfg.separation.backend = "tse"
    cfg.separation.enabled = True
    cfg.tse.model_path = str(tmp_path / "missing.pt")

    pipeline = pipe_mod.build_pipeline(cfg)
    assert isinstance(pipeline.separator, TseSeparator)
    assert pipeline.separator.model is not None

    pipeline.stem_embedder = FakeEcapaEmbedder()
    pipeline.diarizer = FakeTseDiarizer()
    from conversation_deconvolution.conversation.reconstructor import (
        HeuristicReconstructor,
    )
    from conversation_deconvolution.core.config import ReconstructionConfig

    pipeline.reconstructor = HeuristicReconstructor(_NullEmbedder(), ReconstructionConfig())
    pipeline.asr = RecordingAsr()
    audio = np.zeros(3 * SR, dtype=np.float32)
    result = pipeline.run(audio)
    assert len(result.utterances) == 2
