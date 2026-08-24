import json

import pytest

from conversation_deconvolution.audio.loader import load_audio
from conversation_deconvolution.core.config import PipelineConfig, SyntheticConfig
from conversation_deconvolution.synthetic.generator import SyntheticGenerator
from conversation_deconvolution.synthetic.tts import PiperTts


@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    import soundfile as sf

    from conversation_deconvolution.cli import evaluate_results
    from conversation_deconvolution.conversation.export import load_json
    from conversation_deconvolution.core.types import result_from_dict
    from conversation_deconvolution.pipeline import build_pipeline

    gen = SyntheticGenerator(PiperTts(), SyntheticConfig(snr_db=20.0, mean_gap_sec=0.6))
    ds = gen.generate(
        tmp_path / "ds",
        seed=7,
        n_conversations=2,
        speakers_per_thread=2,
        n_lines=(3, 4),
    )
    audio, sr = sf.read(ds / "mixed.wav", dtype="float32")
    assert sr == 16000 and len(audio) > 160000

    cfg = PipelineConfig.default()
    cfg.asr.model_size = "tiny"
    cfg.asr.compute_type = "float32"
    cfg.asr.language = "fr"
    pipeline = build_pipeline(cfg)
    result = pipeline.run(load_audio(ds / "mixed.wav"))

    speakers = {u.speaker for u in result.utterances}
    assert len(result.utterances) >= 3
    assert len(speakers) >= 2 if speakers else True

    gt = result_from_dict(load_json(ds / "ground_truth.json"))
    metrics = evaluate_results(gt, result)
    assert metrics["WER (non-overlap)"] < 1.0
    print(json.dumps(metrics, indent=2))


@pytest.mark.slow
def test_diarizer_separates_two_real_voices(tmp_path):
    from conversation_deconvolution.audio.vad import SileroVad
    from conversation_deconvolution.core.config import DiarizationConfig, VadConfig
    from conversation_deconvolution.diarization.clusterer import AgglomerativeClusterer
    from conversation_deconvolution.diarization.diarizer import SpeakerDiarizer
    from conversation_deconvolution.diarization.embeddings import EcapaEmbedder

    gen = SyntheticGenerator(PiperTts(), SyntheticConfig(mean_gap_sec=0.7))
    ds = gen.generate(
        tmp_path / "ds2",
        seed=5,
        n_conversations=1,
        speakers_per_thread=2,
        n_lines=(3, 3),
    )
    audio = load_audio(ds / "mixed.wav")
    diarizer = SpeakerDiarizer(
        SileroVad(VadConfig()),
        EcapaEmbedder(),
        AgglomerativeClusterer(),
        DiarizationConfig(num_speakers=2),
    )
    turns, _ = diarizer.diarize(audio)
    assert len({t.speaker for t in turns}) == 2
