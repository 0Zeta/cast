"""Validate and serialize the ragged self-splice corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..artifacts import SchemaError


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


# Genuine rows are matched clean twins with no segments.
ROW_KINDS: Final[tuple[str, ...]] = ("blend", "genuine", "ablend",
                                     "a_replace", "a_shift")

# Legacy archives may contain optional a2v/v2a blocks; runtime validation and
# loading ignore them.
CORPUS_KEYS: Final[tuple[str, ...]] = (
    "video_ids", "labels", "kinds", "lengths", "offsets", "seg_offsets",
    "seg_bounds", "scalar", "scalar_b", "scalar_c", "scalar_c_names",
    "sync")

N_SCALAR_COLUMNS: Final[int] = 6

# Per-frame group widths: six scalar tracks per feature space, two artifact
# heads, twelve alignment-residual dimensions.
GROUP_COLUMNS: Final[dict[str, int]] = {
    "scalar": N_SCALAR_COLUMNS, "scalar_b": N_SCALAR_COLUMNS,
    "scalar_c": 2, "sync": 12}

# Archive metadata dtypes; width changes break artifact identity.
METADATA_DTYPES: Final[dict[str, Any]] = {
    "labels": np.int8, "lengths": np.int32, "offsets": np.int64,
    "seg_offsets": np.int64, "seg_bounds": np.int32}

# Artifact-head column tokens, in file order (vart, then aart).
SCALAR_C_NAMES: Final[tuple[str, ...]] = ("vart", "aart")

# Row ids are synthetic and carry no source clip id.
ROW_ID_PATTERN: Final = re.compile(
    r"^(blend|clean|alocal|ablend)[0-9]{5}$")


@dataclass(frozen=True)
class SelfSpliceCorpus:
    """Validated ragged corpus view."""

    video_ids: np.ndarray
    labels: np.ndarray
    kinds: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    seg_offsets: np.ndarray
    seg_bounds: np.ndarray
    scalar: np.ndarray
    scalar_b: np.ndarray
    scalar_c: np.ndarray
    scalar_c_names: tuple[str, ...]
    sync: np.ndarray

    def __len__(self) -> int:
        return len(self.video_ids)

    @classmethod
    def load(cls, path: str | Path) -> "SelfSpliceCorpus":
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in CORPUS_KEYS if k not in z.files]
            _require(
                not missing,
                f"{path}: self-splice corpus is missing keys {missing}")
            c = cls(
                video_ids=z["video_ids"].astype(str),
                labels=z["labels"], kinds=z["kinds"].astype(str),
                lengths=z["lengths"], offsets=z["offsets"],
                seg_offsets=z["seg_offsets"],
                seg_bounds=z["seg_bounds"], scalar=z["scalar"],
                scalar_b=z["scalar_b"], scalar_c=z["scalar_c"],
                scalar_c_names=tuple(
                    str(s) for s in z["scalar_c_names"]),
                sync=z["sync"])
        c.validate(str(path))
        return c

    def validate(self, tag: str = "corpus") -> None:
        n = len(self.video_ids)
        _require(n > 0, f"{tag}: empty corpus")
        _require(len(set(self.video_ids.tolist())) == n,
                 f"{tag}: duplicate row ids")
        bad_ids = [v for v in self.video_ids.tolist()
                   if not ROW_ID_PATTERN.match(v)]
        _require(not bad_ids,
                 f"{tag}: row ids must be synthetic "
                 f"({ROW_ID_PATTERN.pattern}); the corpus carries no "
                 f"source clip id, got {bad_ids[:3]}")
        for name, want in METADATA_DTYPES.items():
            got = getattr(self, name).dtype
            _require(got == want,
                     f"{tag}: {name} must be {np.dtype(want)} (the "
                     f"archive storage dtype), got {got}")
        for name in ("labels", "kinds", "lengths"):
            _require(len(getattr(self, name)) == n,
                     f"{tag}: {name} must have one entry per row")
        unknown = set(self.kinds.tolist()) - set(ROW_KINDS)
        _require(not unknown, f"{tag}: unknown row kinds {sorted(unknown)}")
        genuine = self.kinds == "genuine"
        _require(bool((self.labels[genuine] == 0).all())
                 and bool((self.labels[~genuine] == 1).all()),
                 f"{tag}: labels must be 0 exactly on genuine rows")
        _require(bool((self.lengths >= 1).all()),
                 f"{tag}: nonpositive row length")
        for name, off, total in (
                ("offsets", self.offsets, int(self.lengths.sum())),
                ("seg_offsets", self.seg_offsets, len(self.seg_bounds))):
            _require(len(off) == n + 1 and off[0] == 0
                     and bool((np.diff(off) >= 0).all())
                     and off[-1] == total,
                     f"{tag}: {name} must be a prefix sum ending at "
                     f"{total}")
        _require(bool((np.diff(self.offsets)
                       == self.lengths.astype(np.int64)).all()),
                 f"{tag}: offsets disagree with lengths")
        n_segs = np.diff(self.seg_offsets)
        _require(bool((n_segs[genuine] == 0).all()),
                 f"{tag}: genuine rows must carry no synthetic segment")
        _require(bool((n_segs[~genuine] >= 1).all()),
                 f"{tag}: every synthetic row must carry a segment")
        if len(self.seg_bounds):
            _require(self.seg_bounds.ndim == 2
                     and self.seg_bounds.shape[1] == 2,
                     f"{tag}: seg_bounds must be (n, 2)")
            row = np.repeat(np.arange(n), n_segs)
            _require(bool((self.seg_bounds[:, 0] >= 0).all())
                     and bool((self.seg_bounds[:, 1]
                               >= self.seg_bounds[:, 0]).all())
                     and bool((self.seg_bounds[:, 1]
                               <= self.lengths[row] - 1).all()),
                     f"{tag}: segment bounds outside their row")
        total = int(self.lengths.sum())
        for name in ("scalar", "scalar_b", "scalar_c", "sync"):
            g = getattr(self, name)
            _require(g.ndim == 2 and len(g) == total,
                     f"{tag}: {name} must be ({total}, columns)")
            _require(g.dtype == np.float16,
                     f"{tag}: {name} must be float16 (the archive "
                     f"storage dtype), got {g.dtype}")
        for name, want in GROUP_COLUMNS.items():
            got = getattr(self, name).shape[1]
            _require(got == want,
                     f"{tag}: {name} must have {want} columns, got "
                     f"{got}")
        _require(self.scalar_c.shape[1] == len(self.scalar_c_names),
                 f"{tag}: scalar_c has {self.scalar_c.shape[1]} columns "
                 f"but {len(self.scalar_c_names)} names")
        _require(self.scalar_c_names == SCALAR_C_NAMES,
                 f"{tag}: scalar_c_names must be "
                 f"{list(SCALAR_C_NAMES)} in that order, got "
                 f"{list(self.scalar_c_names)}")

    def row_frames(self, i: int, group: str) -> np.ndarray:
        return getattr(self, group)[self.offsets[i]:self.offsets[i + 1]]

    def row_segments(self, i: int) -> np.ndarray:
        return self.seg_bounds[self.seg_offsets[i]:self.seg_offsets[i + 1]]


def corpus_arrays(video_ids: list[str], labels: list[int],
                  kinds: list[str],
                  groups: dict[str, list[np.ndarray]],
                  segments: list[list[tuple[int, int]]],
                  scalar_c_names: tuple[str, ...],
                  ) -> dict[str, np.ndarray]:
    """Build corpus arrays in archive key order."""
    lengths = np.array([len(s) for s in groups["scalar"]],
                       dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    n_segs = np.array([len(s) for s in segments], dtype=np.int32)
    seg_offsets = np.zeros(len(n_segs) + 1, dtype=np.int64)
    np.cumsum(n_segs, out=seg_offsets[1:])
    seg_rows = [np.asarray(s, dtype=np.int32).reshape(-1, 2)
                for s in segments]
    arrays = {
        "video_ids": np.array(video_ids),
        "labels": np.array(labels, dtype=np.int8),
        "kinds": np.array(kinds),
        "lengths": lengths, "offsets": offsets,
        "seg_offsets": seg_offsets,
        "seg_bounds": (np.concatenate(seg_rows) if n_segs.sum()
                       else np.zeros((0, 2), dtype=np.int32)),
        "scalar": np.concatenate(groups["scalar"]),
        "scalar_b": np.concatenate(groups["scalar_b"]),
        "scalar_c": np.concatenate(groups["scalar_c"]),
        "scalar_c_names": np.array(scalar_c_names),
        "sync": np.concatenate(groups["sync"]),
    }
    # Retain optional legacy batteries in archive order.
    for key in ("a2v", "v2a"):
        if key in groups:
            arrays[key] = np.concatenate(groups[key])
    return arrays


def corpus_from_arrays(arrays: dict[str, np.ndarray]) -> "SelfSpliceCorpus":
    """A validated view over an in-memory corpus layout."""
    missing = [k for k in CORPUS_KEYS if k not in arrays]
    _require(not missing,
             f"corpus arrays are missing keys {missing}")
    c = SelfSpliceCorpus(
        video_ids=arrays["video_ids"].astype(str),
        labels=arrays["labels"], kinds=arrays["kinds"].astype(str),
        lengths=arrays["lengths"], offsets=arrays["offsets"],
        seg_offsets=arrays["seg_offsets"],
        seg_bounds=arrays["seg_bounds"], scalar=arrays["scalar"],
        scalar_b=arrays["scalar_b"], scalar_c=arrays["scalar_c"],
        scalar_c_names=tuple(str(s) for s in arrays["scalar_c_names"]),
        sync=arrays["sync"])
    return c


def write_corpus_npz(path: str | Path, video_ids: list[str],
                     labels: list[int], kinds: list[str],
                     groups: dict[str, list[np.ndarray]],
                     segments: list[list[tuple[int, int]]],
                     scalar_c_names: tuple[str, ...]) -> None:
    """Write the corpus NPZ in the archive layout; the result must
    pass SelfSpliceCorpus.load."""
    np.savez(path, **corpus_arrays(video_ids, labels, kinds, groups,
                                   segments, scalar_c_names))


__all__ = ["ROW_KINDS", "CORPUS_KEYS", "N_SCALAR_COLUMNS",
           "GROUP_COLUMNS", "METADATA_DTYPES", "SCALAR_C_NAMES",
           "ROW_ID_PATTERN", "SelfSpliceCorpus", "corpus_arrays",
           "corpus_from_arrays", "write_corpus_npz", "SchemaError"]
