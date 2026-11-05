#!/usr/bin/env python3
"""Build external-dataset evaluation manifests."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FAV_VOX_INDEX = (
    REPOSITORY_ROOT / "configs" / "cohorts" /
    "fakeavceleb_voxceleb2_paper.csv.gz"
)
FAV_AVPF_SOURCE_INDEX = (
    REPOSITORY_ROOT / "configs" / "cohorts" /
    "fakeavceleb_avpf_released_6667.csv.gz"
)
FAV_AVPF_OMITTED_RELATIVE = (
    "FakeVideo-RealAudio/Asian (South)/women/id04582/"
    "00180_id06445_NWtdqOxEbDY_faceswap.mp4"
)
FAV_AVPF_CATEGORY_COUNTS = {"A": 151, "B": 151, "C": 3_007, "D": 3_358}
FAV_AVPF_CATEGORY_TYPES = {
    "A": "RealVideo-RealAudio",
    "B": "RealVideo-FakeAudio",
    "C": "FakeVideo-RealAudio",
    "D": "FakeVideo-FakeAudio",
}

# video_id namespaces; media access uses video_path instead.
ID_ROOTS = {
    "fakeavceleb": "fakeavceleb",
    "voxceleb2": "voxceleb2",
    "lavdf": "lavdf",
    "avlips": "avlips",
}

OUTPUT_FIELDS = (
    "video_id",
    "video_path",
    "target",
    "modify_type",
    "fake_segments_json",
    "duration",
    "video_frames",
)


def _posix_join(root: str, relative: str) -> str:
    return str(PurePosixPath(root) / PurePosixPath(relative))


def _variable_path(variable: str, relative: str) -> str:
    if not variable.isidentifier():
        raise ValueError(f"invalid environment-variable name {variable!r}")
    return "${" + variable + "}/" + str(PurePosixPath(relative))


def _require_new_output(output: str | Path) -> None:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")


def _write(rows: list[dict[str, object]], output: str | Path) -> None:
    destination = Path(output)
    _require_new_output(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in OUTPUT_FIELDS}
            for row in rows
        )
    print(f"wrote {len(rows):,} rows to {destination}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _gzip_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != fields:
        raise ValueError(
            f"cohort fields in {path} are {reader.fieldnames!r}; "
            f"expected {fields!r}")
    return list(reader)


class _MediaVerifier:
    """Check many paths without issuing one remote-filesystem stat per row."""

    def __init__(self) -> None:
        self._directory_entries: dict[Path, set[str]] = {}

    def exists(self, path: Path) -> bool:
        parent = path.parent
        if parent not in self._directory_entries:
            try:
                self._directory_entries[parent] = {
                    entry.name for entry in os.scandir(parent)
                    if entry.is_file(follow_symlinks=True)
                }
            except FileNotFoundError:
                self._directory_entries[parent] = set()
        return path.name in self._directory_entries[parent]

    def require(self, path: Path) -> None:
        if not self.exists(path):
            raise FileNotFoundError(path)


def _require_counts(
    label: str,
    rows: Iterable[dict[str, object]],
    expected: dict[tuple[int, str], int],
) -> None:
    observed = Counter(
        (int(row["target"]), str(row["modify_type"])) for row in rows
    )
    if observed != Counter(expected):
        raise ValueError(
            f"{label} composition differs: observed={dict(observed)}, "
            f"expected={expected}"
        )


def _fakeavceleb_metadata(root: Path) -> dict[str, tuple[int, str]]:
    metadata_path = root / "meta_data.csv"
    _require_file(metadata_path)
    result: dict[str, tuple[int, str]] = {}
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "type", "path"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(
                f"{metadata_path} lacks columns {sorted(required)}"
            )
        prefix_column = (reader.fieldnames or [""])[-1]
        for row in reader:
            relative = PurePosixPath(str(row[prefix_column])) / str(row["path"])
            parts = relative.parts
            if parts and parts[0] == "FakeAVCeleb":
                relative = PurePosixPath(*parts[1:])
            key = str(relative)
            value = (0 if row["method"] == "real" else 1, str(row["type"]))
            # The upstream metadata lists 22 physical files twice. Their
            # binary target agrees; the paper cohort kept the first row.
            if key in result and result[key][0] != value[0]:
                raise ValueError(f"conflicting FakeAVCeleb target for {key}")
            result.setdefault(key, value)
    return result


def build_fakeavceleb_voxceleb2(args: argparse.Namespace) -> None:
    _require_new_output(args.output)
    fav_root = Path(args.fakeavceleb_root).expanduser().resolve()
    vox_root = Path(args.voxceleb2_root).expanduser().resolve()
    metadata = _fakeavceleb_metadata(fav_root)
    _require_file(FAV_VOX_INDEX)
    media = _MediaVerifier()

    rows: list[dict[str, object]] = []
    with gzip.open(FAV_VOX_INDEX, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "relative_path", "modify_type", "target"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"invalid cohort index: {FAV_VOX_INDEX}")
        for item in reader:
            dataset = str(item["dataset"])
            relative = str(PurePosixPath(str(item["relative_path"])))
            target = int(item["target"])
            modify_type = str(item["modify_type"])
            if dataset == "fakeavceleb":
                local = fav_root / Path(relative)
                id_root = ID_ROOTS[dataset]
                if relative not in metadata:
                    raise ValueError(
                        f"cohort path is absent from FakeAVCeleb metadata: "
                        f"{relative}"
                    )
                if metadata[relative] != (target, modify_type):
                    raise ValueError(
                        f"FakeAVCeleb metadata differs for {relative}: "
                        f"{metadata[relative]} != {(target, modify_type)}"
                    )
            elif dataset == "voxceleb2":
                local = vox_root / Path(relative)
                id_root = ID_ROOTS[dataset]
                if (target, modify_type) != (0, "vox2_real"):
                    raise ValueError(f"invalid VoxCeleb2 cohort row: {item}")
            else:
                raise ValueError(f"unknown cohort dataset {dataset!r}")
            media.require(local)
            rows.append({
                "video_id": _posix_join(id_root, relative),
                "video_path": _variable_path(
                    args.fakeavceleb_path_variable
                    if dataset == "fakeavceleb"
                    else args.voxceleb2_path_variable,
                    relative,
                ),
                "target": target,
                "modify_type": modify_type,
                "fake_segments_json": "[]",
            })

    if len(rows) != 42_060:
        raise ValueError(f"paper cohort has {len(rows)} rows, expected 42,060")
    if len({str(row["video_id"]) for row in rows}) != len(rows):
        raise ValueError("paper cohort has duplicate video_id values")
    _require_counts(
        "FakeAVCeleb+VoxCeleb2",
        rows,
        {
            (0, "vox2_real"): 20_520,
            (0, "RealVideo-RealAudio"): 499,
            (1, "RealVideo-FakeAudio"): 499,
            (1, "FakeVideo-RealAudio"): 9_707,
            (1, "FakeVideo-FakeAudio"): 10_835,
        },
    )
    _write(rows, args.output)


def build_fakeavceleb_avpf(args: argparse.Namespace) -> None:
    """Rebuild CAST's exact available-paper join to AVPF's FAV list."""
    _require_new_output(args.output)
    root = Path(args.fakeavceleb_root).expanduser().resolve()
    metadata = _fakeavceleb_metadata(root)
    source = _gzip_rows(
        FAV_AVPF_SOURCE_INDEX,
        ["source", "category", "full_path"],
    )
    if len(source) != 6_667:
        raise ValueError(
            f"AVPF FakeAVCeleb source has {len(source):,} rows; "
            "expected 6,667")
    source_paths = [str(row["full_path"]) for row in source]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("AVPF FakeAVCeleb source list contains duplicates")
    categories = Counter(str(row["category"]) for row in source)
    if dict(categories) != FAV_AVPF_CATEGORY_COUNTS:
        raise ValueError(
            f"AVPF FakeAVCeleb category counts are {dict(categories)}; "
            f"expected {FAV_AVPF_CATEGORY_COUNTS}")

    _require_file(FAV_VOX_INDEX)
    with gzip.open(FAV_VOX_INDEX, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "relative_path", "modify_type", "target"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"invalid cohort index: {FAV_VOX_INDEX}")
        available = {
            str(item["relative_path"]):
                (str(item["modify_type"]), int(item["target"]))
            for item in reader if item["dataset"] == "fakeavceleb"
        }
    if len(available) != 21_540:
        raise ValueError(
            f"paper FakeAVCeleb availability index has {len(available):,} "
            "rows; expected 21,540")

    media = _MediaVerifier()
    derived: list[tuple[str, str, int]] = []
    omitted: list[tuple[str, str]] = []
    for row_index, item in enumerate(source, start=2):
        full_path = PurePosixPath(str(item["full_path"]))
        if (full_path.is_absolute() or ".." in full_path.parts
                or not full_path.parts or full_path.parts[0] != "FakeAVCeleb"):
            raise ValueError(
                f"AVPF source row {row_index}: invalid full_path "
                f"{str(full_path)!r}")
        relative = str(PurePosixPath(*full_path.parts[1:]))
        category = str(item["category"])
        if category not in FAV_AVPF_CATEGORY_TYPES:
            raise ValueError(
                f"AVPF source row {row_index}: unknown category {category!r}")
        media.require(root / Path(relative))
        if relative not in metadata:
            raise ValueError(
                f"AVPF source row {row_index}: {relative!r} is absent "
                "from official FakeAVCeleb metadata")
        target, modify_type = metadata[relative]
        expected_type = FAV_AVPF_CATEGORY_TYPES[category]
        if modify_type != expected_type:
            raise ValueError(
                f"AVPF source row {row_index}: category {category} implies "
                f"{expected_type}, official metadata says {modify_type}")
        if relative not in available:
            omitted.append((relative, category))
            continue
        if available[relative] != (modify_type, target):
            raise ValueError(
                f"recorded paper availability label differs for {relative}: "
                f"{available[relative]} != {(modify_type, target)}")
        derived.append((relative, modify_type, target))

    if omitted != [(FAV_AVPF_OMITTED_RELATIVE, "C")]:
        raise ValueError(
            "AVPF FakeAVCeleb availability join omitted an unexpected set: "
            f"{omitted}")
    if len(derived) != 6_666:
        raise ValueError(
            f"CAST/AVPF availability join has {len(derived):,} rows; "
            "expected 6,666")

    rows = [{
        "video_id": _posix_join(ID_ROOTS["fakeavceleb"], relative),
        "video_path": _variable_path(args.path_variable, relative),
        "target": target,
        "modify_type": modify_type,
        "fake_segments_json": "[]",
    } for relative, modify_type, target in derived]
    _require_counts(
        "FakeAVCeleb AVPF availability join",
        rows,
        {
            (0, "RealVideo-RealAudio"): 151,
            (1, "RealVideo-FakeAudio"): 151,
            (1, "FakeVideo-RealAudio"): 3_006,
            (1, "FakeVideo-FakeAudio"): 3_358,
        },
    )
    _write(rows, args.output)
    print("excluded one AVPF source row absent from the recorded paper "
          f"FakeAVCeleb availability cohort: {FAV_AVPF_OMITTED_RELATIVE}")


def _lavdf_type(item: dict[str, object]) -> str:
    video = bool(item["modify_video"])
    audio = bool(item["modify_audio"])
    if video and audio:
        return "both"
    if video:
        return "video"
    if audio:
        return "audio"
    return "real"


def build_lavdf(args: argparse.Namespace) -> None:
    _require_new_output(args.output)
    media_root = Path(args.root).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    _require_file(metadata_path)
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, list):
        raise ValueError("LAV-DF metadata must be a JSON list")

    selected = [item for item in metadata if item.get("split") == "test"]
    selected.sort(key=lambda item: Path(str(item["file"])).name)
    rows: list[dict[str, object]] = []
    names: set[str] = set()
    media = _MediaVerifier()
    for item in selected:
        required = {
            "file", "n_fakes", "fake_periods", "duration",
            "modify_video", "modify_audio", "video_frames",
        }
        if not required <= set(item):
            raise ValueError(
                f"LAV-DF metadata row lacks {sorted(required - set(item))}"
            )
        official_relative = PurePosixPath(str(item["file"]))
        name = official_relative.name
        if name in names:
            raise ValueError(f"duplicate LAV-DF test basename: {name}")
        names.add(name)
        nested = media_root / Path(*official_relative.parts)
        flat = media_root / name
        if media.exists(nested):
            local_relative = str(official_relative)
        elif media.exists(flat):
            local_relative = name
        else:
            raise FileNotFoundError(
                f"LAV-DF media is absent at both {nested} and {flat}"
            )
        modify_type = _lavdf_type(item)
        target = int(int(item["n_fakes"]) > 0)
        if target != int(modify_type != "real"):
            raise ValueError(f"inconsistent LAV-DF labels for {name}")
        rows.append({
            "video_id": _posix_join(ID_ROOTS["lavdf"], name),
            "video_path": _variable_path(
                args.path_variable, local_relative
            ),
            "target": target,
            "modify_type": modify_type,
            "fake_segments_json": json.dumps(item["fake_periods"]),
            "duration": item["duration"],
            "video_frames": item["video_frames"],
        })

    if len(rows) != 26_100:
        raise ValueError(f"LAV-DF test has {len(rows)} rows, expected 26,100")
    _require_counts(
        "LAV-DF test",
        rows,
        {
            (0, "real"): 6_906,
            (1, "video"): 6_452,
            (1, "audio"): 6_373,
            (1, "both"): 6_369,
        },
    )
    _write(rows, args.output)


def _numbered_mp4s(directory: Path) -> list[Path]:
    paths = list(directory.glob("*.mp4"))
    if any(not path.stem.isdigit() for path in paths):
        raise ValueError(f"non-numeric MP4 basename in {directory}")
    return sorted(paths, key=lambda path: int(path.stem))


def build_avlips(args: argparse.Namespace) -> None:
    _require_new_output(args.output)
    root = Path(args.root).expanduser().resolve()
    specs = (
        ("0_real", 0, "real", 3_396),
        ("1_fake", 1, "lipsync", 4_206),
    )
    rows: list[dict[str, object]] = []
    for directory, target, modify_type, expected in specs:
        paths = _numbered_mp4s(root / directory)
        if len(paths) != expected:
            raise ValueError(
                f"AVLips {directory} has {len(paths)} MP4s, expected {expected}"
            )
        for path in paths:
            relative = f"{directory}/{path.name}"
            rows.append({
                "video_id": _posix_join(
                    ID_ROOTS["avlips"], relative
                ),
                "video_path": _variable_path(
                    args.path_variable, relative
                ),
                "target": target,
                "modify_type": modify_type,
                "fake_segments_json": "[]",
            })
    if len(rows) != 7_602:
        raise ValueError(f"AVLips has {len(rows)} rows, expected 7,602")
    _write(rows, args.output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="dataset", required=True)

    fav = commands.add_parser(
        "fakeavceleb-voxceleb2",
        help="rebuild the exact 42,060-row paper cohort",
    )
    fav.add_argument(
        "--fakeavceleb-root", required=True,
        help="directory containing FakeAVCeleb meta_data.csv",
    )
    fav.add_argument(
        "--voxceleb2-root", required=True,
        help="VoxCeleb2 dev/mp4 directory",
    )
    fav.add_argument(
        "--fakeavceleb-path-variable", default="FAKEAVCELEB_ROOT",
        help="environment-variable name written into FakeAVCeleb video_path",
    )
    fav.add_argument(
        "--voxceleb2-path-variable", default="VOXCELEB2_ROOT",
        help="environment-variable name written into VoxCeleb2 video_path",
    )
    fav.add_argument("--output", required=True)
    fav.set_defaults(function=build_fakeavceleb_voxceleb2)

    fav_avpf = commands.add_parser(
        "fakeavceleb-avpf",
        help="rebuild the exact 6,666-row CAST/AVPF availability join",
    )
    fav_avpf.add_argument(
        "--fakeavceleb-root", required=True,
        help="directory containing FakeAVCeleb meta_data.csv",
    )
    fav_avpf.add_argument(
        "--path-variable", default="FAKEAVCELEB_ROOT",
        help="environment-variable name written into video_path",
    )
    fav_avpf.add_argument("--output", required=True)
    fav_avpf.set_defaults(function=build_fakeavceleb_avpf)

    lavdf = commands.add_parser(
        "lavdf", help="rebuild the official 26,100-clip LAV-DF test split"
    )
    lavdf.add_argument(
        "--root", required=True,
        help="directory containing the extracted test MP4s",
    )
    lavdf.add_argument(
        "--metadata", required=True, help="official metadata.min.json"
    )
    lavdf.add_argument(
        "--path-variable", default="LAVDF_ROOT",
        help="environment-variable name written into video_path",
    )
    lavdf.add_argument("--output", required=True)
    lavdf.set_defaults(function=build_lavdf)

    avlips = commands.add_parser(
        "avlips", help="rebuild the complete AVLips cohort"
    )
    avlips.add_argument(
        "--root", required=True,
        help="directory containing 0_real/ and 1_fake/",
    )
    avlips.add_argument(
        "--path-variable", default="AVLIPS_ROOT",
        help="environment-variable name written into video_path",
    )
    avlips.add_argument("--output", required=True)
    avlips.set_defaults(function=build_avlips)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
