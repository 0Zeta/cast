"""Download the pretrained encoders and mouth-alignment assets."""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
import shutil
import urllib.request


def download_assets(directory: str | Path, manifest: str | Path,
                    only: list[str] | None = None) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(manifest).read_text())["external_files"]
    known = {record["id"] for record in records}
    if only and set(only) - known:
        raise ValueError(f"unknown assets: {sorted(set(only) - known)}")
    outputs = []
    for record in records:
        if only and record["id"] not in only:
            continue
        destination = directory / record["filename"]
        outputs.append(destination)
        if destination.is_file():
            print(f"using {destination}")
            continue
        temporary = destination.with_name(destination.name + ".part")
        try:
            print(f"downloading {record['filename']}", flush=True)
            with urllib.request.urlopen(record["url"], timeout=120) as response:
                stream = (bz2.BZ2File(response) if
                          record.get("download_compression") == "bzip2" else response)
                with temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--only", nargs="+", help="asset IDs to download")
    parser.add_argument("--manifest", default=str(
        Path(__file__).resolve().parents[3] / "configs/assets.json"))
    args = parser.parse_args(argv)
    download_assets(args.directory, args.manifest, args.only)
