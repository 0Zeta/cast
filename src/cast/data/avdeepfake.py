#!/usr/bin/env python3
"""Build evaluation manifests from AV-Deepfake metadata."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path, PurePosixPath


MAX_CLIP_SEC = 30.0
EXPECTED_ROWS = {
    "avdeepfake1m": 57_340,
    "avdeepfake1m-plus": 77_326,
}
# video_id namespace per dataset; CAST uses video_id only as a join key.
NAMESPACE = {"avdeepfake1m": "avdeepfake1m",
             "avdeepfake1m-plus": "avdeepfake1m_plus"}
COHORTS = Path(__file__).resolve().parents[3] / "configs/cohorts"
# Paper-run sampling identities of the genuine clips, by relative path.
IDENTITY_TABLES = {
    "avdeepfake1m": COHORTS / "avdeepfake1m_sampling_identity.csv.gz",
    "avdeepfake1m-plus": COHORTS / "avdeepfake1m_plus_sampling_identity.csv.gz",
}
IDENTITY_FIELDS = ("pool_split", "calibration_seed", "fusion_seed")
EXPECTED_SILENT_ROWS = 398
SHARED_COHORT_NAME = "avh-align-avpf-10000"
SHARED_COHORT_ROWS = 10_000
SHARED_COHORT_TARGET_COUNTS = {0: 2_495, 1: 7_505}
SHARED_COHORT_PATH = COHORTS / "avdeepfake1m_avh_align_avpf_10000.csv.gz"
BOLDISOR_COHORT_NAME = "boldisor-5133"
BOLDISOR_COHORT_ROWS = 5_133
BOLDISOR_COHORT_TARGET_COUNTS = {0: 2_495, 1: 2_638}
BOLDISOR_COHORT_MODIFY_TYPE_COUNTS = {
    "real": 2_495,
    "audio_modified": 59,
    "visual_modified": 32,
    "both_modified": 2_547,
}
BOLDISOR_COHORT_PATH = COHORTS / "avdeepfake1m_boldisor_5133.csv.gz"
INDEXED_COHORTS = {
    SHARED_COHORT_NAME: {'path': SHARED_COHORT_PATH, 'rows': SHARED_COHORT_ROWS, 'target_counts': SHARED_COHORT_TARGET_COUNTS, 'modify_type_counts': None},
    BOLDISOR_COHORT_NAME: {'path': BOLDISOR_COHORT_PATH, 'rows': BOLDISOR_COHORT_ROWS, 'target_counts': BOLDISOR_COHORT_TARGET_COUNTS, 'modify_type_counts': BOLDISOR_COHORT_MODIFY_TYPE_COUNTS},
}
MODIFY_TYPES = {"real", "audio_modified", "visual_modified",
                "both_modified"}
FIELDS = (
    "video_id", "video_path", "speaker_id", "target", "modify_type",
    "audio_model", "video_model", "source", "video_frames",
    "audio_frames", "clip_sec", "n_fake_segments", "total_fake_sec",
    "fake_segments_json", "audio_fake_segments_json",
    "visual_fake_segments_json", "truncated", "lost_segments_json",
    "lost_fake_sec", "kept_fake_sec", *IDENTITY_FIELDS,
)


def load_identity_table(dataset: str) -> dict[str, dict[str, str]]:
    """relative path -> paper-run sampling identity of a genuine clip."""
    with gzip.open(IDENTITY_TABLES[dataset], "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["relative_path", *IDENTITY_FIELDS]:
            raise ValueError(f"invalid identity table {IDENTITY_TABLES[dataset]}")
        return {row["relative_path"]: {k: row[k] for k in IDENTITY_FIELDS}
                for row in reader}


def _segments(value: object, *, field: str, row_index: int) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"row {row_index}: {field} is not a list")
    result = []
    for segment in value:
        if (not isinstance(segment, list) or len(segment) != 2
                or not all(isinstance(x, (int, float)) for x in segment)):
            raise ValueError(
                f"row {row_index}: invalid {field} segment {segment!r}")
        start, stop = map(float, segment)
        if start < 0 or stop <= start:
            raise ValueError(
                f"row {row_index}: invalid {field} bounds {segment!r}")
        result.append([start, stop])
    return result


def _split_segments(segments: list[list[float]], cap: float):
    lost = []
    kept_sec = 0.0
    lost_sec = 0.0
    for start, stop in segments:
        if start >= cap:
            lost.append([start, stop])
            lost_sec += stop - start
        elif stop > cap:
            lost.append([cap, stop])
            lost_sec += stop - cap
            kept_sec += cap - start
        else:
            kept_sec += stop - start
    return lost, lost_sec, kept_sec


def _speaker(dataset: str, relative: str) -> str:
    parts = PurePosixPath(relative).parts
    if dataset == "avdeepfake1m":
        return parts[0]
    if parts[0] == "vox_celeb_2":
        return parts[1]
    if parts[0] == "lrs3":
        return parts[1]
    return parts[1]  # complete-view silent_videos/<subject>/...


def _load_indexed_cohort(
        metadata_by_file: dict[str, tuple[int, dict]],
        cohort_name: str) -> list[tuple[int, dict]]:
    spec = INDEXED_COHORTS[cohort_name]
    compressed = spec["path"].read_bytes()
    raw = gzip.decompress(compressed)

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != ["path", "label"]:
        raise ValueError(
            f"bundled cohort fields are {reader.fieldnames!r}; "
            "expected ['path', 'label']")
    cohort = list(reader)
    if len(cohort) != spec["rows"]:
        raise ValueError(
            f"bundled {cohort_name} cohort has {len(cohort):,} rows; "
            f"expected {spec['rows']:,}")

    paths = [row["path"] for row in cohort]
    if len(set(paths)) != len(paths):
        raise ValueError("bundled cohort contains duplicate paths")
    selected = []
    target_counts: Counter[int] = Counter()
    modify_type_counts: Counter[str] = Counter()
    for row_index, row in enumerate(cohort, start=2):
        relative = row["path"]
        path = PurePosixPath(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"bundled cohort row {row_index}: invalid path {relative!r}")
        if row["label"] not in {"0", "1"}:
            raise ValueError(
                f"bundled cohort row {row_index}: invalid label "
                f"{row['label']!r}")
        if relative not in metadata_by_file:
            raise ValueError(
                f"bundled cohort row {row_index}: {relative!r} is absent "
                "from official validation metadata")
        metadata_index, record = metadata_by_file[relative]
        target = int(record.get("modify_type") != "real")
        if int(row["label"]) != target:
            raise ValueError(
                f"bundled cohort row {row_index}: label disagrees with "
                f"official metadata for {relative!r}")
        target_counts[target] += 1
        modify_type_counts[str(record["modify_type"])] += 1
        selected.append((metadata_index, record))
    if dict(target_counts) != spec["target_counts"]:
        raise ValueError(
            f"bundled {cohort_name} cohort target counts are "
            f"{dict(target_counts)}; expected {spec['target_counts']}")
    expected_types = spec["modify_type_counts"]
    if expected_types is not None and dict(modify_type_counts) != expected_types:
        raise ValueError(
            f"bundled {cohort_name} modification counts are "
            f"{dict(modify_type_counts)}; expected {expected_types}")

    return selected


def _row(dataset: str, record: dict, row_index: int,
         path_variable: str, identity: dict[str, dict[str, str]]) -> dict:
    relative = str(record.get("file", ""))
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"row {row_index}: invalid relative file {relative!r}")
    if record.get("split") != "val":
        raise ValueError(
            f"row {row_index}: expected split='val', got {record.get('split')!r}")
    modify_type = record.get("modify_type")
    if modify_type not in MODIFY_TYPES:
        raise ValueError(
            f"row {row_index}: invalid modify_type {modify_type!r}")
    frames = record.get("video_frames")
    if not isinstance(frames, int) or frames <= 0:
        raise ValueError(f"row {row_index}: invalid video_frames {frames!r}")

    segments = _segments(record.get("fake_segments"),
                         field="fake_segments", row_index=row_index)
    audio_segments = _segments(record.get("audio_fake_segments"),
                               field="audio_fake_segments",
                               row_index=row_index)
    visual_segments = _segments(record.get("visual_fake_segments"),
                                field="visual_fake_segments",
                                row_index=row_index)
    if modify_type == "real" and (segments or audio_segments or visual_segments):
        raise ValueError(f"row {row_index}: genuine clip has fake segments")

    lost, lost_sec, kept_sec = _split_segments(segments, MAX_CLIP_SEC)
    source = ("vox_celeb_2" if dataset == "avdeepfake1m"
              else path.parts[0])
    sampling = {k: "" for k in IDENTITY_FIELDS}
    if modify_type == "real":
        recorded = identity.get(f"val/{relative}")
        if recorded is None and not relative.startswith("silent_videos/"):
            raise ValueError(
                f"row {row_index}: genuine clip {relative!r} has no recorded "
                "sampling identity")
        sampling.update(recorded or {})
    return {
        "video_id": f"{NAMESPACE[dataset]}/val/{relative}",
        "video_path": "${" + path_variable + f"}}/val/{relative}",
        "speaker_id": _speaker(dataset, relative),
        "target": int(modify_type != "real"),
        "modify_type": modify_type,
        "audio_model": record.get("audio_model"),
        "video_model": record.get("video_model"),
        "source": source,
        "video_frames": frames,
        "audio_frames": record.get("audio_frames"),
        "clip_sec": frames / 25.0,
        "n_fake_segments": len(segments),
        "total_fake_sec": sum((stop - start for start, stop in segments),
                              0.0),
        "fake_segments_json": json.dumps(segments),
        "audio_fake_segments_json": json.dumps(audio_segments),
        "visual_fake_segments_json": json.dumps(visual_segments),
        "truncated": frames / 25.0 > MAX_CLIP_SEC,
        "lost_segments_json": json.dumps(lost),
        "lost_fake_sec": lost_sec,
        "kept_fake_sec": kept_sec,
        **sampling,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        choices=("avdeepfake1m", "avdeepfake1m-plus"))
    parser.add_argument("--metadata", required=True,
                        help="official validation metadata JSON")
    parser.add_argument("--dataset-root", required=True,
                        help="dataset directory containing val/")
    parser.add_argument("--output", required=True, help="output CSV")
    parser.add_argument(
        "--cohort", default="full",
        choices=("full", *INDEXED_COHORTS),
        help="validation cohort to emit; fixed competitor cohorts apply "
             "only to AV-Deepfake1M")
    parser.add_argument(
        "--path-variable",
        help="environment-variable name written into video_path; defaults "
             "to AVDEEPFAKE1M_ROOT or AVDEEPFAKE1MPP_ROOT")
    parser.add_argument(
        "--include-silent-videos", action="store_true",
        help="AV-Deepfake1M++ only: emit the complete 77,326-row official "
             "view, including 398 silent_videos/ clips with no audio stream")
    args = parser.parse_args(argv)

    if args.include_silent_videos and args.dataset != "avdeepfake1m-plus":
        parser.error("--include-silent-videos only applies to avdeepfake1m-plus")
    if args.cohort != "full" and args.dataset != "avdeepfake1m":
        parser.error(
            f"--cohort {args.cohort} only applies to avdeepfake1m")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    root = Path(args.dataset_root)
    if not (root / "val").is_dir():
        raise FileNotFoundError(f"dataset root has no val/ directory: {root}")
    metadata_path = Path(args.metadata)
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, list):
        raise ValueError("validation metadata must be a JSON list")
    expected = EXPECTED_ROWS[args.dataset]
    if len(metadata) != expected:
        raise ValueError(
            f"{args.dataset} metadata has {len(metadata):,} rows; "
            f"expected {expected:,}")

    relative_files = [str(record.get("file", "")) for record in metadata]
    duplicates = [name for name, count in Counter(relative_files).items()
                  if count > 1]
    if duplicates:
        raise ValueError(f"metadata has duplicate files: {duplicates[:5]}")
    metadata_by_file = {
        name: (index, record)
        for index, (name, record) in enumerate(zip(relative_files, metadata))
    }
    silent = sum(name.startswith("silent_videos/")
                 for name in relative_files)
    if args.dataset == "avdeepfake1m-plus" and silent != EXPECTED_SILENT_ROWS:
        raise ValueError(
            f"metadata has {silent} silent_videos/ rows; "
            f"expected {EXPECTED_SILENT_ROWS}")

    path_variable = args.path_variable or (
        "AVDEEPFAKE1M_ROOT" if args.dataset == "avdeepfake1m"
        else "AVDEEPFAKE1MPP_ROOT")
    if not path_variable.isidentifier():
        raise ValueError(f"invalid environment-variable name {path_variable!r}")
    if args.cohort in INDEXED_COHORTS:
        selected = _load_indexed_cohort(metadata_by_file, args.cohort)
        expected_output = INDEXED_COHORTS[args.cohort]["rows"]
    else:
        selected = [
            (index, record) for index, record in enumerate(metadata)
            if (args.include_silent_videos
                or not str(record.get("file", "")).startswith(
                    "silent_videos/"))
        ]
        expected_output = expected
        if (args.dataset == "avdeepfake1m-plus"
                and not args.include_silent_videos):
            expected_output -= EXPECTED_SILENT_ROWS
    if len(selected) != expected_output:
        raise AssertionError(
            f"selected {len(selected):,} rows; expected {expected_output:,}")

    missing = [str(record["file"]) for _, record in selected
               if not (root / "val" / str(record["file"])).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing):,} selected media files are missing under "
            f"{root / 'val'} (first: {preview})")

    identity = load_identity_table(args.dataset)
    rows = [_row(args.dataset, record, index, path_variable, identity)
            for index, record in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["modify_type"] for row in rows)
    order = (f"bundled {args.cohort} cohort order"
             if args.cohort in INDEXED_COHORTS
             else "official metadata order")
    print(f"wrote {len(rows):,} rows in {order} to {output}")
    print("modify_type: " + ", ".join(
        f"{name}={counts[name]:,}" for name in sorted(counts)))
    if args.dataset == "avdeepfake1m-plus":
        action = "included" if args.include_silent_videos else "excluded"
        print(f"{action} {silent} silent_videos/ rows (no audio stream)")


if __name__ == "__main__":
    main()
