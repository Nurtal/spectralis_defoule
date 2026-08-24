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
