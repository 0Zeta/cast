"""Store proposal-aligned boundary cue features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..artifacts import SchemaError, _require

PERCENTILE_RULE = "ecdf"


@dataclass(frozen=True)
class BoundaryCueStore:
    """Per-proposal boundary cue functionals, row-aligned with a prepared
    proposal table. NaN marks a window that could not be measured."""

    start_frame: np.ndarray      # (P,) inclusive frames
    end_frame: np.ndarray        # (P,)
    features: np.ndarray         # (P, n_columns) float64
    column_names: tuple[str, ...]

    KEYS = ("feats", "p_start", "p_end", "colnames", "pct_rule")

    def __len__(self) -> int:
        return len(self.start_frame)

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryCueStore":
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in cls.KEYS if k not in z.files]
            _require(not missing, f"{path}: boundary cue store is missing keys {missing}")
            feats = z["feats"].astype(np.float64)
            names = tuple(str(c) for c in z["colnames"])
            rule = str(z["pct_rule"])
            start = z["p_start"].astype(np.int64)
            end = z["p_end"].astype(np.int64)
        _require(feats.ndim == 2 and feats.shape[1] == len(names),
                 f"{path}: feats shape {feats.shape} does not match "
                 f"{len(names)} column names")
        _require(len(set(names)) == len(names), f"{path}: duplicate column names")
        _require(len(start) == len(feats) and len(end) == len(feats),
                 f"{path}: geometry arrays do not match feats rows")
        _require(rule == PERCENTILE_RULE,
                 f"{path}: percentile rule {rule!r} is not {PERCENTILE_RULE!r}")
        return cls(start_frame=start, end_frame=end, features=feats,
                   column_names=names)

    def save(self, path: str | Path) -> None:
        np.savez(path, feats=self.features.astype(np.float32),
                 p_start=self.start_frame.astype(np.int32),
                 p_end=self.end_frame.astype(np.int32),
                 colnames=np.asarray(self.column_names),
                 pct_rule=np.asarray(PERCENTILE_RULE))

    def columns(self, names: tuple[str, ...], start_frame: np.ndarray,
                end_frame: np.ndarray, *,
                require_measured: bool = True) -> np.ndarray:
        """Select columns for a proposal table whose geometry must match
        this store row for row."""
        if not (np.array_equal(self.start_frame, start_frame)
                and np.array_equal(self.end_frame, end_frame)):
            raise SchemaError(
                "boundary cue store geometry does not match the proposal table; "
                "the store was extracted for a different proposal set")
        missing = [c for c in names if c not in self.column_names]
        if missing:
            raise SchemaError(f"boundary cue store is missing columns {missing}; "
                              f"has {list(self.column_names)}")
        sel = self.features[:, [self.column_names.index(c) for c in names]]
        if require_measured:
            bad = [c for c, col in zip(names, sel.T)
                   if not bool(np.isfinite(col).all())]
            _require(not bad, f"unmeasured values in boundary cue columns {bad}; "
                              "no substitute value may enter scoring")
        return sel
