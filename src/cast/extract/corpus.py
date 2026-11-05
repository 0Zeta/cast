"""Rebuild the self-splice corpus from extracted genuine clips."""


from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cast.corpus import ExtractionRecord, build_self_splice_corpus

from .features import FeatureRecord, iter_feature_records
from .heads import CrossFitArtifactHeads, PredictionHeads
from .scores import derive_scores


FROZEN_SOURCE_ROWS = 3_600
FROZEN_AVAILABLE_SOURCE_ROWS = 3_599
FROZEN_SOURCE_SPEAKERS = 42
FROZEN_UNAVAILABLE_SOURCE_SUFFIX = (
    "id00018/H8unCMlGOMQ/00065/real.mp4")
FROZEN_CORPUS_COUNTS = {
    "blend": 3_599,
    "genuine": 3_599,
    "a_replace": 2_119,
    "a_shift": 1_479,
    "ablend": 3_599,
}
FROZEN_CORPUS_ROWS = sum(FROZEN_CORPUS_COUNTS.values())


def _extraction_record(record: FeatureRecord) -> ExtractionRecord:
    return ExtractionRecord(
        video_id=record.video_id,
        audio=record.audio,
        visual=record.visual,
        joint=record.joint,
        segments=record.segments,
    )


def _record_map(directory: str | Path,
                *, require_segments: bool = False):
    result = {}
    for record in iter_feature_records(directory):
        converted = _extraction_record(record)
        if require_segments and not converted.segments:
            continue
        result[record.video_id] = converted
    return result


def _clean_map(directory: str | Path):
    return {
        record.video_id: (record.audio, record.visual, record.joint)
        for record in iter_feature_records(directory)
    }


def _speaker_map(manifest: str | Path) -> tuple[dict[str, str], set[str]]:
    result = {}
    unavailable = set()
    with Path(manifest).open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("video_id", "")).strip()
            speaker = str(row.get("speaker_id", "")).strip()
            if not video_id or not speaker:
                raise ValueError(
                    f"{manifest} must provide nonempty video_id and "
                    "speaker_id for every self-splice source")
            if "target" in row and str(row["target"]).strip() not in {"", "0"}:
                raise ValueError(
                    f"{manifest} marks self-splice source {video_id!r} "
                    f"with non-genuine target {row['target']!r}")
            if video_id in result:
                raise ValueError(
                    f"{manifest} contains duplicate video_id {video_id!r}")
            result[video_id] = speaker
            available = str(row.get("frozen_available", "1")).strip()
            if available not in {"0", "1"}:
                raise ValueError(
                    f"{manifest} has invalid frozen_available={available!r} "
                    f"for {video_id!r}")
            if available == "0":
                unavailable.add(video_id)
    return result, unavailable


def _validate_full_source_manifest(
        speakers: dict[str, str], unavailable: set[str]) -> None:
    if len(speakers) != FROZEN_SOURCE_ROWS:
        raise ValueError(
            f"identity build needs {FROZEN_SOURCE_ROWS:,} source rows; "
            f"the manifest has {len(speakers):,}")
    if len(set(speakers.values())) != FROZEN_SOURCE_SPEAKERS:
        raise ValueError(
            f"identity build needs {FROZEN_SOURCE_SPEAKERS} source "
            f"speakers; the manifest has {len(set(speakers.values()))}")
    if len(unavailable) != 1 or not next(iter(unavailable)).endswith(
            FROZEN_UNAVAILABLE_SOURCE_SUFFIX):
        raise ValueError(
            "identity build needs exactly the recorded unavailable "
            f"source */{FROZEN_UNAVAILABLE_SOURCE_SUFFIX}; got "
            f"{sorted(unavailable)}")
    if len(set(speakers) - unavailable) != FROZEN_AVAILABLE_SOURCE_ROWS:
        raise AssertionError("recorded available-source count is inconsistent")


def _validate_full_corpus(record: dict, output: Path) -> dict[str, object]:
    counts = record.get("counts")
    if counts != FROZEN_CORPUS_COUNTS:
        raise ValueError(
            "self-splice composition differs from the recorded corpus: "
            f"observed {counts}, expected {FROZEN_CORPUS_COUNTS}")
    if len(record.get("rows", ())) != FROZEN_CORPUS_ROWS:
        raise ValueError(
            f"full self-splice corpus has {len(record.get('rows', ())):,} "
            f"rows; expected {FROZEN_CORPUS_ROWS:,}")
    if record.get("skipped_pairs") != 0 \
            or record.get("skipped_audio_blends") != 0:
        raise ValueError(
            "full self-splice build lost an available six-leg counterpart: "
            f"skipped_pairs={record.get('skipped_pairs')}, "
            "skipped_audio_blends="
            f"{record.get('skipped_audio_blends')}")

    return {"counts": dict(counts)}


class _CorpusScorer:
    def __init__(self, prediction: PredictionHeads,
                 artifacts: CrossFitArtifactHeads,
                 speakers: dict[str, str]):
        self.prediction = prediction
        self.artifacts = artifacts
        self.speakers = speakers

    def __call__(self, source_video_id, audio, visual, joint,
                 audio_avsr, visual_avsr, joint_avsr):
        if source_video_id not in self.speakers:
            raise KeyError(
                f"self-splice source {source_video_id!r} has no speaker_id")
        speaker = self.speakers[source_video_id]
        prediction = self.prediction.scores(
            audio.astype(np.float16), visual.astype(np.float16))
        return {
            "scalar": derive_scores(audio, visual, joint),
            "scalar_b": derive_scores(
                audio_avsr, visual_avsr, joint_avsr),
            "scalar_c": self.artifacts.scores(
                speaker, audio, visual),
            # Corpus scoring rounds the input feature rows back
            # to fp16 before applying the multivariate head.
            "sync": prediction["sync"],
            "a2v": prediction["a2v"],
            "v2a": prediction["v2a"],
        }


def build_corpus_from_features(
        primary_clean: str | Path, auxiliary_clean: str | Path,
        primary_visual_blend: str | Path,
        auxiliary_visual_blend: str | Path,
        primary_audio_blend: str | Path,
        auxiliary_audio_blend: str | Path,
        source_manifest: str | Path,
        prediction_checkpoint: str | Path,
        visual_crossfit: tuple[str | Path, str | Path],
        audio_crossfit: tuple[str | Path, str | Path],
        output: str | Path, *, device: str = "cuda", limit: int = 0,
        log=print) -> dict:
    """Build the corpus from six feature directories.

    Full builds require the reference source selection and row composition."""
    output = Path(output)
    if limit < 0:
        raise ValueError("self-splice limit must be nonnegative")
    if output.suffix != ".npz":
        raise ValueError("self-splice output path must end in .npz")
    if output.exists() or output.with_suffix(".build.json").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    speakers, unavailable = _speaker_map(source_manifest)
    if not limit:
        _validate_full_source_manifest(speakers, unavailable)
    prediction = PredictionHeads(prediction_checkpoint, device)
    artifacts = CrossFitArtifactHeads(
        visual_crossfit, audio_crossfit, device, salt="s5")
    scorer = _CorpusScorer(prediction, artifacts, speakers)

    log("loading clean and auxiliary blend feature rows")
    clean = _clean_map(primary_clean)
    clean_avsr = _clean_map(auxiliary_clean)
    visual_avsr = _record_map(
        auxiliary_visual_blend, require_segments=True)
    audio_avsr = _record_map(
        auxiliary_audio_blend, require_segments=True)
    for mapping in (clean, clean_avsr, visual_avsr, audio_avsr):
        for video_id in unavailable:
            mapping.pop(video_id, None)

    visual = (_extraction_record(record)
              for record in iter_feature_records(primary_visual_blend)
              if record.segments and record.video_id not in unavailable)
    audio = (_extraction_record(record)
             for record in iter_feature_records(primary_audio_blend)
             if record.segments and record.video_id not in unavailable)
    record_path = output.with_suffix(".build.json")
    try:
        record = build_self_splice_corpus(
            visual,
            visual_avsr,
            clean,
            clean_avsr,
            scorer,
            output,
            audio_blend=audio,
            audio_blend_avsr=audio_avsr,
            seed=0,
            limit=limit,
            log=log,
        )
        if not limit:
            _validate_full_corpus(record, output)
    except BaseException:
        if not limit:
            output.unlink(missing_ok=True)
            record_path.unlink(missing_ok=True)
        raise

    inputs = {
        "source_manifest": str(Path(source_manifest).resolve()),
        "frozen_unavailable_video_ids": sorted(unavailable),
        "feature_directories": {
            "primary_clean": str(Path(primary_clean).resolve()),
            "auxiliary_clean": str(Path(auxiliary_clean).resolve()),
            "primary_visual_blend": str(
                Path(primary_visual_blend).resolve()),
            "auxiliary_visual_blend": str(
                Path(auxiliary_visual_blend).resolve()),
            "primary_audio_blend": str(
                Path(primary_audio_blend).resolve()),
            "auxiliary_audio_blend": str(
                Path(auxiliary_audio_blend).resolve()),
        },
        "checkpoints": {
            "prediction_heads": Path(prediction_checkpoint).name,
            "visual_artifact_crossfit": [Path(p).name for p in visual_crossfit],
            "audio_artifact_crossfit": [Path(p).name for p in audio_crossfit],
        },
    }
    record["inputs"] = inputs

    record_path.write_text(
        json.dumps(record, indent=1) + "\n")
    return record
