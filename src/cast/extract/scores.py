"""Compute and store CAST's four runtime score caches."""


from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


from .features import iter_feature_records

SCORE_NAMES = ("cos", "vcont", "acont", "jv", "ja", "laggain")
MAX_LAG = 3


def derive_scores(audio: np.ndarray, visual: np.ndarray,
                  joint: np.ndarray) -> np.ndarray:
    """Return six scalar tracks with manipulation oriented downward."""
    lengths = {len(audio), len(visual), len(joint)}
    if len(lengths) != 1:
        raise ValueError(
            f"audio/visual/joint lengths disagree: "
            f"{len(audio)}/{len(visual)}/{len(joint)}")

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    a, v, j = normalize(audio), normalize(visual), normalize(joint)
    length = len(a)
    cos = (a * v).sum(1)
    vcont = np.empty(length, np.float32)
    acont = np.empty(length, np.float32)
    if length > 1:
        vcont[1:] = (v[1:] * v[:-1]).sum(1)
        acont[1:] = (a[1:] * a[:-1]).sum(1)
        vcont[0] = vcont[1]
        acont[0] = acont[1]
    elif length == 1:
        vcont[0] = acont[0] = 1.0
    jv = (j * v).sum(1)
    ja = (j * a).sum(1)
    best = cos.copy()
    for lag in range(1, MAX_LAG + 1):
        best[:-lag] = np.maximum(
            best[:-lag], (a[:-lag] * v[lag:]).sum(1))
        best[lag:] = np.maximum(
            best[lag:], (a[lag:] * v[:-lag]).sum(1))
    laggain = cos - best
    return np.stack([cos, vcont, acont, jv, ja, laggain], axis=1)


def _ragged(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]:
    lengths = np.asarray([len(x) for x in sequences], dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if not sequences:
        raise ValueError("no complete feature records were found")
    return lengths, offsets, np.concatenate(sequences)


def _write_scalar(path: Path, video_ids: list[str],
                  sequences: list[np.ndarray], names: tuple[str, ...]) -> None:
    lengths, offsets, scores = _ragged(sequences)
    np.savez(path, video_ids=np.asarray(video_ids), lengths=lengths,
             offsets=offsets, scores=scores,
             score_names=np.asarray(names))


def _write_prediction(path: Path, video_ids: list[str],
                      groups: dict[str, list[np.ndarray]]) -> None:
    lengths, offsets, sync = _ragged(groups["sync"])
    flat = {"sync": sync}
    for key in ("a2v", "v2a"):
        _, group_offsets, values = _ragged(groups[key])
        if not np.array_equal(group_offsets, offsets):
            raise ValueError(f"prediction group {key} has different lengths")
        flat[key] = values
    np.savez(path, video_ids=np.asarray(video_ids), lengths=lengths,
             offsets=offsets, **flat,
             group_dims=json.dumps({
                 key: int(value.shape[1]) for key, value in flat.items()
             }))


def build_runtime_caches(
        primary_features: str | Path,
        auxiliary_features: str | Path,
        prediction_checkpoint: str | Path,
        visual_artifact_checkpoint: str | Path,
        audio_artifact_checkpoint: str | Path,
        output_dir: str | Path,
        *, prefix: str = "evaluation", device: str = "cuda",
        log: Callable[[str], None] = print) -> dict:
    """Write main, AVSR, sync, and artifact caches from primary and auxiliary features.

    Retains unused prediction blocks for archive identity."""
    from .heads import (PredictionHeads, artifact_scores,
                        load_artifact_head)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "main": out / f"{prefix}_main.npz",
        "avsr": out / f"{prefix}_avsr.npz",
        "sync": out / f"{prefix}_sync.npz",
        "artifact": out / f"{prefix}_artifact.npz",
        "record": out / f"{prefix}_cache_build.json",
    }
    collisions = [str(p) for p in paths.values() if p.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing cache outputs: "
            + ", ".join(collisions))

    prediction = PredictionHeads(prediction_checkpoint, device)
    visual_head = load_artifact_head(
        "visual_artifact_head", visual_artifact_checkpoint, device)
    audio_head = load_artifact_head(
        "audio_artifact_head", audio_artifact_checkpoint, device)

    primary_ids: list[str] = []
    main: list[np.ndarray] = []
    prediction_scores: dict[str, list[np.ndarray]] = {
        "sync": [], "a2v": [], "v2a": []}
    artifact: list[np.ndarray] = []
    for index, record in enumerate(iter_feature_records(primary_features), 1):
        primary_ids.append(record.video_id)
        main.append(derive_scores(record.audio, record.visual, record.joint))
        predicted = prediction.scores(record.audio, record.visual)
        for key in prediction_scores:
            prediction_scores[key].append(predicted[key])
        vart = artifact_scores(visual_head, record.visual, device)
        aart = artifact_scores(audio_head, record.audio, device)
        artifact.append(np.stack([vart, aart], axis=1))
        if index % 100 == 0:
            log(f"primary: {index} clips")

    auxiliary_ids: list[str] = []
    avsr: list[np.ndarray] = []
    for index, record in enumerate(iter_feature_records(auxiliary_features), 1):
        auxiliary_ids.append(record.video_id)
        avsr.append(derive_scores(record.audio, record.visual, record.joint))
        if index % 100 == 0:
            log(f"auxiliary: {index} clips")

    _write_scalar(paths["main"], primary_ids, main, SCORE_NAMES)
    _write_scalar(paths["avsr"], auxiliary_ids, avsr, SCORE_NAMES)
    _write_prediction(paths["sync"], primary_ids, prediction_scores)
    _write_scalar(paths["artifact"], primary_ids, artifact,
                  ("vart", "aart"))

    record = {
        "schema_version": 1,
        "primary_features": str(Path(primary_features).resolve()),
        "auxiliary_features": str(Path(auxiliary_features).resolve()),
        "clips": {
            "primary": len(primary_ids),
            "auxiliary": len(auxiliary_ids),
            "common": len(set(primary_ids) & set(auxiliary_ids)),
        },
        "checkpoints": {
            "prediction_heads": {
                "path": str(Path(prediction_checkpoint).resolve()),
            },
            "visual_artifact_head": {
                "path": str(Path(visual_artifact_checkpoint).resolve()),
            },
            "audio_artifact_head": {
                "path": str(Path(audio_artifact_checkpoint).resolve()),
            },
        },
        "outputs": {key: str(value.resolve())
                    for key, value in paths.items() if key != "record"},
        "score_names": {
            "main": list(SCORE_NAMES),
            "avsr": list(SCORE_NAMES),
            "artifact": ["vart", "aart"],
            "sync": ["sync", "a2v", "v2a"],
        },
    }
    paths["record"].write_text(json.dumps(record, indent=2) + "\n")
    log(f"wrote four caches for {len(primary_ids)} primary and "
        f"{len(auxiliary_ids)} auxiliary clips")
    return record
