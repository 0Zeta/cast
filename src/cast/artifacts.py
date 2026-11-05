"""Validate and load CAST CSV and NPZ artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Frame f covers [f / FPS, (f + 1) / FPS) seconds.
FPS: Final[float] = 25.0


class SchemaError(ValueError):
    """An artifact does not match its declared schema."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _video_ids(values: pd.Series, table: str) -> np.ndarray:
    vids = values.to_numpy(dtype=object)
    bad = [repr(v) for v in vids
           if not isinstance(v, str) or not v]
    _require(not bad,
             f"{table} video_id values must be nonempty strings; "
             f"found {bad[:5]}")
    return vids


@dataclass(frozen=True)
class ProposalTable:
    """Ranked proposals with inclusive frame bounds."""

    video_id: np.ndarray
    start_frame: np.ndarray
    end_frame: np.ndarray
    score: np.ndarray

    COLUMNS = ("video_id", "start_frame", "end_frame", "score")

    def __len__(self) -> int:
        return len(self.video_id)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ProposalTable":
        missing = [c for c in cls.COLUMNS if c not in df.columns]
        _require(not missing, f"proposal table is missing columns {missing}")
        start = df["start_frame"].to_numpy()
        end = df["end_frame"].to_numpy()
        try:
            score = df["score"].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SchemaError("proposal score must be numeric") from exc
        _require(np.issubdtype(start.dtype, np.integer)
                 and np.issubdtype(end.dtype, np.integer),
                 "start_frame/end_frame must be integers")
        _require(bool((start >= 0).all()), "negative start_frame")
        _require(bool((end >= start).all()), "end_frame < start_frame")
        _require(bool(np.isfinite(score).all()), "non-finite score")
        return cls(video_id=_video_ids(df["video_id"], "proposal table"),
                   start_frame=start.astype(np.int64),
                   end_frame=end.astype(np.int64),
                   score=score)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ProposalTable":
        return cls.from_dataframe(pd.read_csv(path))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"video_id": self.video_id,
                             "start_frame": self.start_frame,
                             "end_frame": self.end_frame,
                             "score": self.score})

    def to_csv(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)


@dataclass(frozen=True)
class LabelTable:
    """Forgery segments by video_id; target-zero rows always have no segments."""

    video_id: np.ndarray
    segments: tuple[np.ndarray, ...]  # per clip, shape (n_segments, 2)

    def __len__(self) -> int:
        return len(self.video_id)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "LabelTable":
        missing = [c for c in ("video_id", "fake_segments_json")
                   if c not in df.columns]
        _require(not missing, f"label table is missing columns {missing}")
        vids = _video_ids(df["video_id"], "label table")
        _require(len(set(vids)) == len(vids),
                 "duplicate video_id in label table")
        raw = df["fake_segments_json"].tolist()
        if "target" in df.columns:
            tgt = df["target"].to_numpy()
            unknown = sorted({repr(t) for t in tgt.tolist()
                              if t not in (0, 1)})
            _require(not unknown,
                     f"label table target column must be 0 or 1; found "
                     f"{unknown[:5]}")
            raw = [("[]" if t == 0 else r) for r, t in zip(raw, tgt)]
        segs = []
        for vid, r in zip(vids, raw):
            try:
                decoded = json.loads(r)
                s = np.asarray(decoded, dtype=np.float64)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise SchemaError(
                    f"{vid}: fake_segments_json is not a numeric JSON "
                    "array") from exc
            if s.size == 0:
                s = s.reshape(0, 2)
            _require(s.ndim == 2 and s.shape[1] == 2,
                     f"{vid}: segments must be [start_s, end_s] pairs")
            _require(bool(np.isfinite(s).all()),
                     f"{vid}: non-finite segment bound")
            _require(bool((s[:, 0] >= 0).all()),
                     f"{vid}: negative segment start")
            _require(bool((s[:, 1] > s[:, 0]).all()),
                     f"{vid}: zero- or negative-width segment")
            segs.append(s)
        return cls(video_id=vids, segments=tuple(segs))

    @classmethod
    def from_csv(cls, path: str | Path) -> "LabelTable":
        return cls.from_dataframe(pd.read_csv(path))


@dataclass(frozen=True)
class FrameScores:
    """Flat frame scores addressed by per-clip offsets."""

    video_ids: np.ndarray
    offsets: np.ndarray
    frame_scores: np.ndarray

    KEYS = ("video_ids", "offsets", "frame_scores")

    @classmethod
    def load(cls, path: str | Path) -> "FrameScores":
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in cls.KEYS if k not in z.files]
            _require(not missing, f"{path}: frame scores are missing keys "
                                  f"{missing}")
            raw_vids = z["video_ids"]
            raw_off = z["offsets"]
            raw_scores = z["frame_scores"]
        _require(raw_vids.ndim == 1 and raw_vids.dtype.kind in "US",
                 f"{path}: video_ids must be a one-dimensional string array")
        vids = raw_vids.astype(str)
        _require(bool((vids != "").all()),
                 f"{path}: empty video_id")
        _require(len(set(vids.tolist())) == len(vids),
                 f"{path}: duplicate video_id")
        _require(raw_off.ndim == 1
                 and np.issubdtype(raw_off.dtype, np.integer),
                 f"{path}: offsets must be a one-dimensional integer array")
        if np.issubdtype(raw_off.dtype, np.unsignedinteger) and raw_off.size:
            _require(int(raw_off.max()) <= np.iinfo(np.int64).max,
                     f"{path}: offsets exceed int64 range")
        off = raw_off.astype(np.int64)
        _require(raw_scores.ndim == 1
                 and np.issubdtype(raw_scores.dtype, np.number)
                 and not np.issubdtype(raw_scores.dtype,
                                      np.complexfloating),
                 f"{path}: frame_scores must be a one-dimensional numeric "
                 "array")
        sc = raw_scores.astype(np.float32)
        _require(bool(np.isfinite(sc).all()),
                 f"{path}: non-finite frame score")
        _require(len(off) == len(vids) + 1,
                 f"{path}: offsets must have len(video_ids) + 1 entries")
        _require(off[0] == 0 and bool((np.diff(off) >= 0).all())
                 and off[-1] == len(sc),
                 f"{path}: offsets must be nondecreasing from 0 to "
                 f"len(frame_scores)")
        return cls(video_ids=vids, offsets=off, frame_scores=sc)

    def clip(self, i: int) -> np.ndarray:
        return self.frame_scores[self.offsets[i]:self.offsets[i + 1]]
