import json

import numpy as np
import pytest
import soundfile as sf

from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
from conversation_deconvolution.conversation.semantic import SentenceTransformerEmbedder
from conversation_deconvolution.core.config import PipelineConfig, SyntheticConfig
from conversation_deconvolution.synthetic.generator import SyntheticGenerator


class FakeTts:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice):
        self.calls.append((text, voice))
        dur = 0.3 + 0.01 * len(text.split())
        n = int(dur * 16000)
        t = np.arange(n) / 16000
        freq = 200 + hash(voice) % 300
        return (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32"), 16000


def _make_fake_pipeline():
    text_embedder = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
    from conversation_deconvolution.diarization.vad import SileroVad

    from conversation_deconvolution.asr.faster_whisper_asr import FasterWhisperAsr
    from conversation_deconvolution.core.config import ReconstructionConfig
    from conversation_deconvolution.diarization.clusterer import AgglomerativeClusterer
    from conversation_deconvolution.diarization.diarizer import SpeakerDiarizer
    from conversation_deconvolution.diarization.embeddings import EcapaEmbedder
    from conversation_deconvolution.pipeline import DeconvolutionPipeline
    from conversation_deconvolution.separation.passthrough import PassthroughSeparator

    vad = SileroVad(PipelineConfig().vad)
    embedder = EcapaEmbedder()
    clusterer = AgglomerativeClusterer(PipelineConfig().diarization.distance_threshold)
    diarizer = SpeakerDiarizer(vad, embedder, clusterer, PipelineConfig().diarization)
    separator = PassthroughSeparator()
    asr = FasterWhisperAsr(PipelineConfig().asr)

    return DeconvolutionPipeline(
        diarizer=diarizer,
        separator=separator,
        asr=asr,
        reconstructor=HeuristicReconstructor(text_embedder, ReconstructionConfig()),
        config=PipelineConfig(),
    )


class TestRobustnessPhase7:
    """Phase 7: Robustness - tester le système dans des conditions réalistes."""

    @pytest.mark.parametrize("snr_db", [5, 10, 15, 20])
    def test_snr_conditions_generator(self, tmp_path, snr_db):
        """Évaluer la génération de dataset selon le SNR."""
        cfg = SyntheticConfig(sample_rate=16000, snr_db=snr_db)
        gen = SyntheticGenerator(FakeTts(), cfg)
        out = gen.generate(
            tmp_path / "ds", seed=42, n_conversations=2, speakers_per_thread=2, n_lines=(3, 5)
        )
        # Vérifier que le fichier audio existe
        assert (out / "mixed.wav").exists()
        # Vérifier la vérité terrain
        import json

        gt = json.load(open(out / "ground_truth.json"))
        assert len(gt["conversations"]) == 2

    @pytest.mark.parametrize("n_speakers", [2, 3, 4])
    def test_speaker_count_generator(self, tmp_path, n_speakers):
        """Tester la génération avec différents nombres de locuteurs."""
        cfg = SyntheticConfig(sample_rate=16000, snr_db=15)
        gen = SyntheticGenerator(FakeTts(), cfg)
        out = gen.generate(
            tmp_path / "ds",
            seed=42,
            n_conversations=2,
            speakers_per_thread=n_speakers,
            n_lines=(3, 5),
        )
        assert (out / "mixed.wav").exists()
        gt = json.load(open(out / "ground_truth.json"))
        # Vérifier que le bon nombre de speakers est généré
        all_speakers = set()
        for conv in gt["conversations"]:
            all_speakers.update(conv["participants"])
        assert len(all_speakers) >= n_speakers

    @pytest.mark.parametrize("max_gap", [10.0, 30.0])
    def test_reconstruction_gap_config(self, max_gap):
        """Tester la configuration du gap maximum pour la reconstruction."""
        from conversation_deconvolution.core.config import ReconstructionConfig

        cfg = ReconstructionConfig(max_gap=max_gap)
        assert cfg.max_gap == max_gap

    @pytest.mark.parametrize("snr_db", [5, 10, 15])
    def test_snr_audio_quality(self, tmp_path, snr_db):
        """Vérifier la qualité audio générée à différents SNR."""
        cfg = SyntheticConfig(sample_rate=16000, snr_db=snr_db)
        gen = SyntheticGenerator(FakeTts(), cfg)
        out = gen.generate(
            tmp_path / "ds", seed=42, n_conversations=1, speakers_per_thread=1, n_lines=(3, 3)
        )
        assert (out / "mixed.wav").exists()
        data, sr = sf.read(out / "mixed.wav")
        assert sr == 16000
        # Vérifier que l'audio n'est pas silencieux
        assert np.max(np.abs(data)) > 0.01

    @pytest.mark.parametrize("n_conversations", [1, 2, 3])
    def test_n_conversations_generator(self, tmp_path, n_conversations):
        """Tester la génération avec différents nombres de conversations."""
        cfg = SyntheticConfig(sample_rate=16000, snr_db=15)
        gen = SyntheticGenerator(FakeTts(), cfg)
        out = gen.generate(
            tmp_path / "ds",
            seed=42,
            n_conversations=n_conversations,
            speakers_per_thread=2,
            n_lines=(3, 5),
        )
        assert (out / "mixed.wav").exists()
        gt = json.load(open(out / "ground_truth.json"))
        assert len(gt["conversations"]) == n_conversations

    def test_deterministic_seed(self, tmp_path):
        """Vérifier que la génération est reproductible avec le même seed."""
        cfg = SyntheticConfig(sample_rate=16000, snr_db=15)
        gen = SyntheticGenerator(FakeTts(), cfg)
        out1 = gen.generate(
            tmp_path / "ds1",
            seed=123,
            n_conversations=2,
            speakers_per_thread=2,
            n_lines=(3, 5),
        )
        out2 = gen.generate(
            tmp_path / "ds2",
            seed=123,
            n_conversations=2,
            speakers_per_thread=2,
            n_lines=(3, 5),
        )
        import json

        gt1 = json.load(open(out1 / "ground_truth.json"))
        gt2 = json.load(open(out2 / "ground_truth.json"))
        assert gt1 == gt2
