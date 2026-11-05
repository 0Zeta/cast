"""Load and validate CAST configuration."""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval.localization import (AP_IOU_THRESHOLDS, AR_IOU_GRID,
                                AR_PROPOSAL_COUNTS)

# Candidate channels per scalar score group and per multivariate group, in
# construction order.
SCALAR_CHANNELS: tuple[str, ...] = ("energy", "drop", "edge", "step_contrast")
MULTIVARIATE_CHANNELS: tuple[str, ...] = ("energy", "edge")
CANDIDATE_FUSIONS = ("learned", "equal")
CLIP_BASELINES = ("clip_specific", "pooled")


def _channel_subset(values: tuple[str, ...], canonical: tuple[str, ...],
                    label: str) -> None:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be a nonempty list of distinct channels")
    unknown = [v for v in values if v not in canonical]
    if unknown:
        raise ValueError(f"{label}: unknown channels {unknown}; expected a "
                         f"subset of {list(canonical)}")
    if tuple(v for v in canonical if v in values) != tuple(values):
        raise ValueError(f"{label} must follow the order {list(canonical)}")


@dataclass(frozen=True)
class ScanConfig:
    """Interval-scan and calibration settings."""
    guard_band_frames: int = 0
    min_interval_frames: int = 25
    max_duration_frames: int = 127
    edge_margin_frames: int = 8
    top_k: int = 50
    nms_iou: float = 0.5
    min_cell_references: int = 100
    min_tail_exceedances: int = 30
    clip_min_references: int = 50
    insertion_rate: float = 0.05
    n_folds: int = 4
    #: candidate channels of every scalar score group
    scalar_channels: tuple[str, ...] = SCALAR_CHANNELS
    #: candidate channels of every multivariate group
    multivariate_channels: tuple[str, ...] = MULTIVARIATE_CHANNELS
    #: learned nonnegative candidate-fusion weights, or equal weights
    candidate_fusion: str = "learned"
    #: clip-specific baseline (held-out-rank clip effect), or the pooled
    #: baseline that drops the clip effect and keeps its prior variance
    clip_baseline: str = "clip_specific"

    def __post_init__(self) -> None:
        _channel_subset(self.scalar_channels, SCALAR_CHANNELS, "scalar_channels")
        _channel_subset(self.multivariate_channels, MULTIVARIATE_CHANNELS,
                        "multivariate_channels")
        if self.candidate_fusion not in CANDIDATE_FUSIONS:
            raise ValueError(f"candidate_fusion must be one of {CANDIDATE_FUSIONS}")
        if self.clip_baseline not in CLIP_BASELINES:
            raise ValueError(f"clip_baseline must be one of {CLIP_BASELINES}")


@dataclass(frozen=True)
class RankingConfig:
    """Proposal ranking defaults."""
    min_proposal_width: int = 3


@dataclass(frozen=True)
class EvalConfig:
    """Metric grids and bootstrap settings."""
    ap_iou_thresholds: tuple[float, ...] = AP_IOU_THRESHOLDS
    ar_proposal_counts: tuple[int, ...] = AR_PROPOSAL_COUNTS
    ar_iou_grid: tuple[float, ...] = AR_IOU_GRID
    bootstrap_resamples: int = 20000
    bootstrap_seed: int = 0
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        for label, values, fmt in (
                ("ap_iou_thresholds", self.ap_iou_thresholds,
                 lambda v: f"ap@{v:g}"),
                ("ar_proposal_counts", self.ar_proposal_counts,
                 lambda v: f"ar@{v}")):
            keys = [fmt(v) for v in values]
            if len(set(keys)) != len(keys):
                raise ValueError(
                    f"{label} {list(values)} do not give distinct "
                    f"metric keys {keys}")


@dataclass(frozen=True)
class ExecutionConfig:
    nproc: int = 8
    seed: int = 0


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Validate a value against its dataclass annotation without bool/int coercion."""

    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        rest = [a for a in args if a is not type(None)]
        hint, origin = rest[0], typing.get_origin(rest[0])
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{where}: expected a list, got "
                f"{type(value).__name__} {value!r}")
        args = typing.get_args(hint)
        if not args:
            return tuple(value)
        return tuple(_coerce(v, args[0], f"{where}[{i}]")
                     for i, v in enumerate(value))
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"{where}: expected a bool, got "
                f"{type(value).__name__} {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{where}: expected an int, got "
                f"{type(value).__name__} {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{where}: expected a number, got "
                f"{type(value).__name__} {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(
                f"{where}: expected a string, got "
                f"{type(value).__name__} {value!r}")
        return value
    return value


def _section(cls: type, d: dict[str, Any], name: str) -> Any:
    if not isinstance(d, dict):
        raise TypeError(
            f"config section '{name}' must be an object, got "
            f"{type(d).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise KeyError(
            f"unknown config keys {sorted(unknown)} in section '{name}'; "
            f"unrecognised configuration is refused rather than ignored")
    hints = typing.get_type_hints(cls)
    return cls(**{k: _coerce(v, hints[k], f"{name}.{k}")
                  for k, v in d.items()})


@dataclass(frozen=True)
class CastConfig:
    scan: ScanConfig = ScanConfig()
    ranking: RankingConfig = RankingConfig()
    evaluation: EvalConfig = EvalConfig()
    execution: ExecutionConfig = ExecutionConfig()

    _SECTIONS = (("scan", ScanConfig), ("ranking", RankingConfig),
                 ("evaluation", EvalConfig), ("execution", ExecutionConfig))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CastConfig":
        if not isinstance(d, dict):
            raise TypeError(
                f"configuration must be an object, got {type(d).__name__}")
        names = [n for n, _ in cls._SECTIONS]
        unknown = set(d) - set(names)
        if unknown:
            raise KeyError(
                f"unknown config sections {sorted(unknown)}; expected a "
                f"subset of {names}")
        return cls(**{n: _section(c, d.get(n, {}), n)
                      for n, c in cls._SECTIONS})

    @classmethod
    def load(cls, path: str | Path) -> "CastConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True)
                     + "\n")


def load_config_document(path: str | Path) -> CastConfig:
    """Load a method configuration."""
    raw = json.loads(Path(path).read_text())
    return CastConfig.from_dict(raw)
