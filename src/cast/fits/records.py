"""Save fitted duration models and their scientific settings."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ..corpus import DEPLOYED_PLACEMENT_LAW
from .duration_law import fit_duration_law
from .pausal_units import fit_max_duration_frames
from .words import MIN_CONTENT_WORDS, NON_LEXICAL_LABELS, load_word_intervals

SUPERVISION_GENUINE_DERIVED = "genuine-derived"
REGISTERED_MAX_DURATION_BAND = (60, 250)


def build_fit_record(object_name: str, supervision_level: str,
                     training_population: str, fit: dict[str, Any],
                     inputs: dict[str, str | Path],
                     notes: str | None = None) -> dict[str, Any]:
    paths = {name: Path(path) for name, path in inputs.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    record = {"object": object_name, "supervision_level": supervision_level,
              "training_population": training_population, "fit": fit,
              "inputs": {name: str(path) for name, path in paths.items()}}
    if notes:
        record["notes"] = notes
    return record


def write_fit_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def read_fit_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def duration_law_record(words_csv: str | Path) -> dict[str, Any]:
    fit = fit_duration_law(load_word_intervals(words_csv))
    payload = {
        "rule": "mean and population SD of log content-word duration in "
                "25-fps frames; cap = floor(p99)",
        **dataclasses.asdict(fit),
        "fit_quotation_4dp": fit.fit_quotation_4dp,
        "deployed_constants": dataclasses.asdict(DEPLOYED_PLACEMENT_LAW),
        "min_content_words": MIN_CONTENT_WORDS,
        "non_lexical_labels": sorted(NON_LEXICAL_LABELS),
    }
    return build_fit_record(
        "content_word_duration_law", SUPERVISION_GENUINE_DERIVED,
        f"VoxCeleb2-dev: {fit.n_clips} genuine clips, {fit.n_words} content words",
        payload, {"word_alignments_csv": words_csv})


def max_duration_record(words_csv: str | Path) -> dict[str, Any]:
    fit = fit_max_duration_frames(load_word_intervals(words_csv))
    lo, hi = REGISTERED_MAX_DURATION_BAND
    if not lo <= fit.max_duration_frames <= hi:
        raise ValueError(
            f"max_duration_frames={fit.max_duration_frames} is outside "
            f"[{lo}, {hi}] (p99={fit.p99_frames:.4f}, n_units={fit.n_units})")
    payload = {
        "rule": "p99 of genuine interior inter-pausal-unit durations in "
                "25-fps frames; successive word gaps below 200 ms",
        **dataclasses.asdict(fit),
        "min_content_words": MIN_CONTENT_WORDS,
    }
    return build_fit_record(
        "max_duration_frames", SUPERVISION_GENUINE_DERIVED,
        f"VoxCeleb2-dev: {fit.n_units} interior units over {fit.n_clips} genuine clips",
        payload, {"word_alignments_csv": words_csv})
