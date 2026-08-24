import json
from pathlib import Path

import numpy as np

from conversation_deconvolution.core.types import (
    TranscriptResult,
    result_to_dict,
)


def save_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False))


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_result(result: TranscriptResult, path: str | Path) -> None:
    save_json(path, result_to_dict(result))
