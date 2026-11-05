"""Scientific settings for the head-training command."""

from __future__ import annotations

import json
import math
from pathlib import Path


DEFAULT_HEAD_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "heads.json"

_SETTING_TYPES = {
    "seed": int,
    "learning_rate": float,
    "steps_align": int,
    "steps_ref": int,
    "steps_pred": int,
    "batch_size": int,
    "window_frames": int,
    "validation_batches": int,
    "pca_train_batches": int,
    "pca_validation_batches": int,
    "prediction_validation_batches": int,
    "epochs": int,
    "preload": bool,
    "augment": bool,
}
_NONNEGATIVE = {"seed", "steps_align", "steps_ref", "steps_pred"}


def validate_head_config(settings: dict) -> None:
    """Reject unknown fields, coercions, and invalid scalar training values."""
    if not isinstance(settings, dict):
        raise TypeError("head configuration must be an object")
    unknown = settings.keys() - _SETTING_TYPES.keys()
    if unknown:
        raise ValueError(f"unknown head configuration settings: {sorted(unknown)}")
    for name, value in settings.items():
        expected = _SETTING_TYPES[name]
        if type(value) is not expected:
            raise TypeError(f"{name} must be a {expected.__name__}")
        if expected is bool:
            continue
        if expected is float:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        elif name in _NONNEGATIVE:
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        elif value <= 0:
            raise ValueError(f"{name} must be positive")


def _read_settings(path: Path) -> dict:
    settings = json.loads(path.read_text())
    validate_head_config(settings)
    return settings


def load_head_config(path: str | Path | None = None) -> dict:
    """Load repository defaults and an optional partial file relative to CWD."""
    settings = _read_settings(DEFAULT_HEAD_CONFIG)
    missing = _SETTING_TYPES.keys() - settings.keys()
    if missing:
        raise ValueError(f"missing default head configuration settings: {sorted(missing)}")
    if path is not None:
        settings.update(_read_settings(Path(path)))
    return settings
