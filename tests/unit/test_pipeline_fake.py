import json

import numpy as np

from conversation_deconvolution.core.config import PipelineConfig
from conversation_deconvolution.core.types import (
    Segment,
    SpeakerTurn,
    VadResult,
    result_from_dict,
)
from conversation_deconvolution.pipeline import DeconvolutionPipeline


class FakeVad:
    def detect(self, audio):
        return VadResult(
            segments=[Segment(0.0, 1.0), Segment(1.2, 2.2)],
            frame_probs=np.zeros(500),
            frame_rate=50.0,
        )


class FakeDiarizer:
    def __init__(self):
        self.vad = FakeVad()

    def diarize(self, audio):
        turns = [
            SpeakerTurn("SPEAKER_00", 0.0, 1.0),
            SpeakerTurn("SPEAKER_01", 0.4, 1.4),
            SpeakerTurn("SPEAKER_00", 2.0, 3.0),
        ]
        return turns, [np.ones(8) for _ in turns]


class FakeSeparator:
    def separate(self, mix, regions):
        from conversation_deconvolution.core.types import SeparationResult

        return SeparationResult(mix=np.asarray(mix).copy())


class FakeAsr:
    def transcribe(self, segment, language=None):
        n = len(segment)
        text = "bonjour le cafe du matin" if n > 20000 else "oui je viens au cafe"
        import types

        return types.SimpleNamespace(text=text, confidence=0.9, language="fr")


class FakeTextEmbedder:
    def encode(self, texts):
        dim = 32
        out = []
        for t in texts:
            v = np.zeros(dim)
            for w in t.split():
                v[hash(w) % dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n)
        return np.array(out)


def make_pipeline():
    from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
    from conversation_deconvolution.core.config import ReconstructionConfig

    return DeconvolutionPipeline(
        diarizer=FakeDiarizer(),
        separator=FakeSeparator(),
        asr=FakeAsr(),
        reconstructor=HeuristicReconstructor(FakeTextEmbedder(), ReconstructionConfig()),
        config=PipelineConfig(),
    )


def test_pipeline_fake_end_to_end():
    audio = np.zeros(48000, dtype=np.float32)
    result = make_pipeline().run(audio)
    assert len(result.utterances) == 3
    assert {u.speaker for u in result.utterances} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(u.text for u in result.utterances)
    assert result.overlaps and result.overlaps[0] == Segment(0.4, 1.0)
    assert len(result.conversations) >= 1
    assert sum(len(c.utterances) for c in result.conversations) == 3


def test_pipeline_result_roundtrip(tmp_path):
    audio = np.zeros(48000, dtype=np.float32)
    result = make_pipeline().run(audio)
    p = tmp_path / "out.json"
    from conversation_deconvolution.conversation.export import save_result

    save_result(result, p)
    loaded = result_from_dict(json.loads(p.read_text()))
    key = lambda us: [(u.speaker, u.start, u.end, u.text) for u in us]
    assert key(loaded.utterances) == key(result.utterances)
    assert len(loaded.conversations) == len(result.conversations)
