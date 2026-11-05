"""Read sharded AV-HuBERT feature records."""


from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FeatureRecord:
    """One complete AV-HuBERT feature record."""

    video_id: str
    audio: np.ndarray
    visual: np.ndarray
    joint: np.ndarray
    metadata: dict

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        raw = (self.metadata.get("segments")
               or self.metadata.get("blend_segs")
               or self.metadata.get("blend_segs_a")
               or ())
        return tuple((int(a), int(b)) for a, b in raw)


def feature_parts(directory: str | Path) -> list[Path]:
    """Return feature tar parts in deterministic order."""
    root = Path(directory)
    parts = sorted(root.glob("shard_*_part_*.tar"))
    if not parts:
        parts = sorted(root.glob("shard_*.tar"))
    if not parts:
        raise FileNotFoundError(f"no feature tar shards in {root}")
    return parts


def iter_feature_records(directory: str | Path,
                         *, tolerate_truncated: bool = False,
                         deduplicate: bool = True,
                         ) -> Iterator[FeatureRecord]:
    """Yield complete records in lexical part order, keeping the first duplicate."""
    seen: set[str] = set()
    for part in feature_parts(directory):
        pending: dict[str, dict] = {}
        try:
            with tarfile.open(part) as archive:
                for member in archive:
                    name = member.name
                    suffix = next((s for s in (
                        ".json", ".audio.npy", ".visual.npy", ".joint.npy"
                    ) if name.endswith(s)), None)
                    if suffix is None:
                        continue
                    key = name[:-len(suffix)]
                    slot = pending.setdefault(key, {})
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    if suffix == ".json":
                        slot["metadata"] = json.load(handle)
                    else:
                        slot[suffix] = np.load(io.BytesIO(handle.read()),
                                               allow_pickle=False)
                    if len(slot) != 4:
                        continue
                    complete = pending.pop(key)
                    meta = complete["metadata"]
                    video_id = str(meta["video_id"])
                    if deduplicate and video_id in seen:
                        continue
                    seen.add(video_id)
                    yield FeatureRecord(
                        video_id=video_id,
                        audio=complete[".audio.npy"],
                        visual=complete[".visual.npy"],
                        joint=complete[".joint.npy"],
                        metadata=meta,
                    )
        except (tarfile.ReadError, EOFError, OSError):
            if not tolerate_truncated:
                raise
