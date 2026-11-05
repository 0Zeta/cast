#!/usr/bin/env python3
"""Select self-splice sources from an ordered genuine manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cast.corpus import source_manifest_indices

REFERENCE_UNAVAILABLE_SUFFIX = (
    "id00018/H8unCMlGOMQ/00065/real.mp4")


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _train_id_root(rows: list[dict]) -> str:
    roots = set()
    for row in rows:
        video_id = str(row["video_id"])
        marker = "/train/"
        if marker not in video_id:
            raise ValueError(
                f"video_id {video_id!r} has no /train/ namespace boundary")
        roots.add(video_id.split(marker, 1)[0])
    if len(roots) != 1:
        raise ValueError(
            f"genuine manifest mixes train video_id roots: {sorted(roots)}")
    return roots.pop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genuine-manifest", required=True,
                        help="ordered CSV with video_id, video_path, speaker_id")
    parser.add_argument("--train-metadata", required=True,
                        help="official AV-Deepfake1M train_metadata.json")
    parser.add_argument("--sources-out", required=True)
    parser.add_argument("--reference-exclusion-out", required=True,
                        help="all genuine rows from the selected speakers")
    args = parser.parse_args(argv)

    outputs = [Path(args.sources_out), Path(args.reference_exclusion_out)]
    if outputs[0].resolve() == outputs[1].resolve():
        parser.error("--sources-out and --reference-exclusion-out must differ")
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing outputs: " + ", ".join(existing))

    source = Path(args.genuine_manifest)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    required = {"video_id", "video_path", "speaker_id"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{source} is missing columns {sorted(missing)}")
    if any(not str(row["speaker_id"]).strip() for row in rows):
        raise ValueError(f"{source} contains blank speaker_id values")
    if len(rows) != 45_063:
        raise ValueError(
            f"{source} has {len(rows)} rows; the paper source selection "
            "expects the 45,063-row genuine-reference manifest")
    id_root = _train_id_root(rows)

    selected = [rows[index] for index in source_manifest_indices(len(rows))]
    if "frozen_available" not in fields:
        fields.append("frozen_available")
    for row in selected:
        row["frozen_available"] = int(
            not str(row["video_id"]).endswith(REFERENCE_UNAVAILABLE_SUFFIX))
    if sum(int(row["frozen_available"]) for row in selected) != 3_599:
        raise ValueError(
            "the unavailable paper-run source was not found exactly once; "
            "the manifest is keyed or ordered differently")
    speakers = {row["speaker_id"] for row in selected}
    metadata_bytes = Path(args.train_metadata).read_bytes()
    metadata = json.loads(metadata_bytes)
    exclusion = [
        {"video_id": f"{id_root}/train/{row['file']}"}
        for row in metadata
        if row.get("modify_type") == "real"
        and str(row["file"]).split("/", 1)[0] in speakers
    ]
    if len(exclusion) != 5_237:
        raise ValueError(
            f"source-speaker exclusion has {len(exclusion)} rows; "
            "the paper-run AV-Deepfake1M metadata yields 5,237")
    _write(Path(args.sources_out), selected, fields)
    # CAST consumes only video_id here; retaining no paths keeps the exclusion
    # artifact portable and prevents machine-local roots from leaking into it.
    _write(Path(args.reference_exclusion_out), exclusion, ["video_id"])
    print(f"selected {len(selected)} source clips (3,599 available in the "
          f"paper extraction) from {len(speakers)} speakers; exclusion "
          f"contains {len(exclusion)} clips")


if __name__ == "__main__":
    main()
