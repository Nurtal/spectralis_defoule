import numpy as np
import pytest

from conversation_deconvolution.core.config import PipelineConfig


def test_defaults():
    cfg = PipelineConfig.default()
    assert cfg.reconstruction.max_gap == 30.0
    assert cfg.vad.threshold == 0.5
    assert cfg.diarization.num_speakers is None


def test_default_loads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("vad:\n  threshold: 0.6\ndiarization:\n  num_speakers: 4\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.vad.threshold == 0.6
    assert cfg.diarization.num_speakers == 4
    assert cfg.asr.model_size == "small"


def test_separation_section_loads(tmp_path):
    p = tmp_path / "sep.yaml"
    p.write_text("separation:\n  enabled: true\n  assign_min_sim: 0.4\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.separation.enabled is True
    assert cfg.separation.assign_min_sim == 0.4
    assert cfg.separation.model_name == "speechbrain/sepformer-whamr16k"


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("vad:\n  nonsense: 1\n")
    with pytest.raises(TypeError):
        PipelineConfig.from_yaml(p)


def test_graph_defaults():
    cfg = PipelineConfig.default()
    assert cfg.reconstructor_kind == "heuristic"
    assert cfg.graph.model_path == "models/graph_lr.json"
    assert cfg.graph.negative_ratio == 3.0


def test_graph_section_loads(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "graph:\n  edge_threshold: 0.6\n  resolution: 1.2\nreconstructor_kind: graph\n"
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.graph.edge_threshold == 0.6
    assert cfg.graph.resolution == 1.2
    assert cfg.reconstructor_kind == "graph"


def test_repo_default_yaml_loads():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cfg = PipelineConfig.from_yaml(root / "configs" / "default.yaml")
    assert cfg.asr.device == "cuda"
    assert cfg.synthetic.sample_rate == 16000


def test_asr_beam_and_prompt_defaults():
    cfg = PipelineConfig.default()
    assert cfg.asr.beam_size == 1
    assert cfg.asr.language == "fr"
    assert cfg.asr.initial_prompt is None


def test_asr_beam_and_prompt_from_yaml(tmp_path):
    p = tmp_path / "asr.yaml"
    p.write_text(
        "asr:\n  beam_size: 8\n  initial_prompt: 'Conversation technique en francais.'\n"
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.asr.beam_size == 8
    assert cfg.asr.initial_prompt == "Conversation technique en francais."


def test_tse_section_loads(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("tse:\n  model_path: models/tse/model.pt\n  lr: 0.0001\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.tse.model_path == "models/tse/model.pt"
    assert cfg.tse.lr == 0.0001


def test_tse_defaults():
    cfg = PipelineConfig.default()
    assert cfg.tse.n_fft == 512
    assert cfg.tse.lr == 3e-4
    assert cfg.tse.epochs == 30
    assert cfg.tse.model_path == "models/tse/model.pt"


def test_faster_whisper_asr_passes_beam_and_prompt(monkeypatch):
    from conversation_deconvolution.asr.faster_whisper_asr import FasterWhisperAsr
    from conversation_deconvolution.core.config import AsrConfig

    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return iter([]), type("Info", (), {"language": "fr"})()

    cfg = AsrConfig(beam_size=7, initial_prompt="test prompt")
    asr = FasterWhisperAsr(cfg)
    asr._model = FakeModel()
    asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert captured["beam_size"] == 7
    assert captured["initial_prompt"] == "test prompt"


def test_separation_backend_loads(tmp_path):
    p = tmp_path / "sep2.yaml"
    p.write_text("separation:\n  backend: tse\n  enabled: true\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.separation.backend == "tse"
    assert cfg.separation.enabled is True


def test_separation_backend_default():
    cfg = PipelineConfig.default()
    assert cfg.separation.backend == "sepformer"
    assert cfg.separation.enabled is False


def test_tse_config_new_fields():
    from conversation_deconvolution.core.config import TseConfig

    cfg = TseConfig()
    assert cfg.freq_bands == 32
    assert cfg.lambda_rec == 0.5
    assert cfg.lambda_sim == 0.5
