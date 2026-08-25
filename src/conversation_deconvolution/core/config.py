from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100


@dataclass
class DiarizationConfig:
    distance_threshold: float = 0.75
    min_segment_sec: float = 0.4
    num_speakers: int | None = None
    window_sec: float = 1.5
    hop_sec: float = 0.5
    min_turn_sec: float = 0.3


@dataclass
class SeparationConfig:
    enabled: bool = False
    model_name: str = "speechbrain/sepformer-whamr16k"
    device: str | None = None
    min_region_sec: float = 0.4
    pad_sec: float = 0.25
    assign_min_sim: float = 0.35
    assign_min_margin: float = 0.03


@dataclass
class AsrConfig:
    model_size: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "fr"
    context_pad_sec: float = 0.0
    beam_size: int = 1
    initial_prompt: str | None = None


@dataclass
class ReconstructionConfig:
    max_gap: float = 30.0
    tau: float = 4.0
    w_temporal: float = 0.25
    w_alternation: float = 0.15
    w_semantic: float = 0.45
    w_same_speaker: float = 0.55
    threshold: float = 0.5
    max_overlap_ratio: float = 0.5
    max_speaker_overlap_ratio: float = 0.05
    max_successors: int = 2


@dataclass
class SyntheticConfig:
    sample_rate: int = 16000
    snr_db: float = 15.0
    mean_gap_sec: float = 0.8
    min_words: int = 3
    max_words: int = 14


@dataclass
class GraphConfig:
    model_path: str = "models/graph_lr.json"
    max_gap: float = 30.0
    tau: float = 4.0
    edge_threshold: float = 0.5
    resolution: float = 1.0
    seed: int = 0
    negative_ratio: float = 3.0


@dataclass
class PipelineConfig:
    vad: VadConfig = field(default_factory=VadConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    text_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reconstructor_kind: str = "heuristic"

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"invalid config: {path}")
        sections = {
            "vad": VadConfig,
            "diarization": DiarizationConfig,
            "separation": SeparationConfig,
            "asr": AsrConfig,
            "reconstruction": ReconstructionConfig,
            "synthetic": SyntheticConfig,
            "graph": GraphConfig,
        }
        kwargs: dict = {}
        for key, value in raw.items():
            sub = sections.get(key)
            if sub is not None and isinstance(value, dict):
                kwargs[key] = sub(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
