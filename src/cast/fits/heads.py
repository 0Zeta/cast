"""Locate the heads used for score extraction and controlled corruptions."""

from pathlib import Path

AUXILIARY_HEADS = (
    "prediction_heads", "visual_artifact_head", "audio_artifact_head",
)
CORPUS_CROSSFIT_HEADS = (
    "visual_artifact_crossfit_0", "visual_artifact_crossfit_1",
    "audio_artifact_crossfit_0", "audio_artifact_crossfit_1",
)


def head_checkpoint(name: str, path: str | Path) -> Path:
    """Check the head role and return its checkpoint path."""
    if name not in (*AUXILIARY_HEADS, *CORPUS_CROSSFIT_HEADS):
        raise KeyError(f"unknown auxiliary head {name!r}")
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{name} checkpoint: {checkpoint}")
    return checkpoint
