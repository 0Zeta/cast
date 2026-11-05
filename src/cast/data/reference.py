#!/usr/bin/env python3
"""Build the ordered AV-Deepfake1M genuine-reference manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path


NAMESPACE = "avdeepfake1m"
IDENTITY_TABLE = (Path(__file__).resolve().parents[3]
                  / "configs/cohorts/avdeepfake1m_sampling_identity.csv.gz")
IDENTITY_FIELDS = ("pool_split", "calibration_seed", "fusion_seed")
REFERENCE_TARGET_CLIPS = 45_000
REFERENCE_SELECTED_CLIPS = 45_063
REFERENCE_SELECTED_SPEAKERS = 518


def _speaker(relative_path: str) -> str:
    match = re.match(r"(id\d+)/", relative_path)
    return match.group(1) if match else relative_path.split("/", 1)[0]


def _clip_key(path: str) -> str:
    match = re.search(r"(id\d+)/([^/]+)/(\d+)/", path)
    return "/".join(match.groups()) if match else path


def _validation_keys(paths: list[str]) -> set[str]:
    keys = set()
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = (row.get("video_path") or row.get("video_id")
                         or row.get("file") or "")
                if value:
                    keys.add(_clip_key(str(value)))
    return keys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", required=True,
                        help="official AV-Deepfake1M train_metadata.json")
    parser.add_argument("--exclude-validation-manifest", action="append",
                        required=True, metavar="CSV",
                        help="repeat for AV-Deepfake1M and ++ validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-clips", type=int,
                        default=REFERENCE_TARGET_CLIPS)
    parser.add_argument("--path-variable", default="AVDEEPFAKE1M_ROOT",
                        help="environment variable written into video_path")
    args = parser.parse_args(argv)

    if len(args.exclude_validation_manifest) != 2:
        parser.error(
            "pass exactly two --exclude-validation-manifest files: "
            "AV-Deepfake1M and AV-Deepfake1M++ validation")
    if not args.path_variable.isidentifier():
        parser.error(
            f"invalid environment-variable name {args.path_variable!r}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata_bytes = Path(args.train_metadata).read_bytes()
    metadata = json.loads(metadata_bytes)
    validation = _validation_keys(args.exclude_validation_manifest)
    genuine = [row for row in metadata
               if row.get("modify_type") == "real"
               and _clip_key(row["file"]) not in validation]

    by_speaker = {}
    for row in genuine:
        by_speaker.setdefault(_speaker(row["file"]), []).append(row)
    selected_speakers = []
    count = 0
    for speaker in sorted(by_speaker):
        if count >= args.target_clips:
            break
        selected_speakers.append(speaker)
        count += len(by_speaker[speaker])
    selected = [row for speaker in selected_speakers
                for row in by_speaker[speaker]]
    if args.target_clips == REFERENCE_TARGET_CLIPS:
        if (len(selected) != REFERENCE_SELECTED_CLIPS
                or len(selected_speakers) != REFERENCE_SELECTED_SPEAKERS):
            raise ValueError(
                "validation exclusions do not reconstruct the recorded "
                "reference cohort: selected "
                f"{len(selected):,} clips from {len(selected_speakers):,} "
                f"speakers, expected {REFERENCE_SELECTED_CLIPS:,} from "
                f"{REFERENCE_SELECTED_SPEAKERS:,}")

    with gzip.open(IDENTITY_TABLE, "rt", newline="") as handle:
        identity = {row["relative_path"]: row for row in csv.DictReader(handle)}
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("video_id", "video_path", "speaker_id", "target",
              "modify_type", "video_frames", *IDENTITY_FIELDS)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            relative = str(row["file"])
            recorded = identity.get(f"train/{relative}")
            if recorded is None:
                raise ValueError(
                    f"{relative!r} has no recorded sampling identity")
            writer.writerow({
                "video_id": f"{NAMESPACE}/train/{relative}",
                "video_path": "${" + args.path_variable
                              + f"}}/train/{relative}",
                "speaker_id": _speaker(relative),
                "target": 0,
                "modify_type": "real",
                "video_frames": row.get("video_frames", ""),
                **{k: recorded[k] for k in IDENTITY_FIELDS},
            })
    print(f"wrote {len(selected)} clips from {len(selected_speakers)} "
          f"complete speakers to {output}")


if __name__ == "__main__":
    main()
