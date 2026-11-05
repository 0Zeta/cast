"""Memory-mapped frozen features for neural-head training."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from cast.extract.features import feature_parts


def _records(directory, with_joint=False):
    suffixes = (".json", ".audio.npy", ".visual.npy")
    if with_joint:
        suffixes += (".joint.npy",)
    seen = set()
    for part in feature_parts(directory):
        pending = {}
        with tarfile.open(part) as archive:
            for member in archive:
                suffix = next((s for s in suffixes if member.name.endswith(s)), None)
                if suffix is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                key = member.name[:-len(suffix)]
                record = pending.setdefault(key, {})
                record[suffix] = (json.load(handle) if suffix == ".json" else
                                   np.load(io.BytesIO(handle.read()), allow_pickle=False))
                if len(record) != len(suffixes):
                    continue
                pending.pop(key)
                video_id = str(record[".json"]["video_id"])
                if video_id not in seen:
                    seen.add(video_id)
                    yield video_id, record
        if pending:
            raise ValueError(f"incomplete training features in {part}")


def build_feature_cache(tar_dir, out_prefix, with_joint=False) -> Path:
    """Stream lexical TAR records into the original ragged fp16 cache format."""
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    streams = ("audio", "visual", "joint") if with_joint else ("audio", "visual")
    video_ids, lengths = [], []
    dimensions = {}
    for video_id, record in _records(tar_dir, with_joint):
        length = len(record[".audio.npy"])
        for stream in streams:
            values = record[f".{stream}.npy"]
            if values.ndim != 2 or len(values) != length or length == 0:
                raise ValueError(f"invalid {stream} feature shape for {video_id}: {values.shape}")
            dimension = values.shape[1]
            if stream in dimensions and dimensions[stream] != dimension:
                raise ValueError(f"inconsistent {stream} feature dimension")
            dimensions[stream] = dimension
        video_ids.append(video_id)
        lengths.append(length)
    if not video_ids:
        raise ValueError(f"no training features in {tar_dir}")
    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    arrays = {
        stream: np.lib.format.open_memmap(
            f"{prefix}_{stream}.npy", mode="w+", dtype=np.float16,
            shape=(int(offsets[-1]), dimensions[stream]))
        for stream in streams
    }
    count = 0
    for i, (video_id, record) in enumerate(_records(tar_dir, with_joint)):
        if i >= len(video_ids) or video_id != video_ids[i]:
            raise ValueError("training feature records changed while building the cache")
        for stream, array in arrays.items():
            array[offsets[i]:offsets[i + 1]] = record[f".{stream}.npy"]
        count += 1
    if count != len(video_ids):
        raise ValueError("training feature records changed while building the cache")
    for array in arrays.values():
        array.flush()
    np.savez(f"{prefix}_meta.npz", video_ids=np.asarray(video_ids),
             lengths=lengths, offsets=offsets)
    return prefix


class FeatureCache:
    """Read a flat cache prefix, or stage TAR features in a temporary cache."""

    def __init__(self, prefix):
        self._temporary = None
        prefix = Path(prefix)
        if prefix.is_dir():
            self._temporary = tempfile.TemporaryDirectory(prefix="cast-training-")
            prefix = build_feature_cache(prefix, Path(self._temporary.name) / "features")
        with np.load(f"{prefix}_meta.npz", allow_pickle=False) as metadata:
            self.video_ids = metadata["video_ids"]
            self.lengths = metadata["lengths"]
            self.offsets = metadata["offsets"]
        self.audio = np.load(f"{prefix}_audio.npy", mmap_mode="r")
        self.visual = np.load(f"{prefix}_visual.npy", mmap_mode="r")
        joint_path = Path(f"{prefix}_joint.npy")
        self.joint = np.load(joint_path, mmap_mode="r") if joint_path.exists() else None

    def __len__(self):
        return len(self.video_ids)

    def get(self, index):
        start, end = self.offsets[index:index + 2]
        return self.audio[start:end], self.visual[start:end]

    def get3(self, index):
        start, end = self.offsets[index:index + 2]
        return (*self.get(index), None if self.joint is None else self.joint[start:end])
