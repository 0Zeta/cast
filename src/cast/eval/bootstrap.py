"""Compute speaker-clustered bootstrap intervals for localization metrics."""


from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .localization import (MatchTable, ap_from_multiplicities,
                           ar_from_multiplicities)

_VOX_SPEAKER = re.compile(r"^id\d+$")
_LRS3_CORPUS = "lrs3"

# below this share of clips per key, the split is not a real clustering
_MAX_CLUSTERS_PER_CLIP = 0.9
_MAX_SINGLETON_SHARE = 0.99


def speaker_cluster_key(video_id: str) -> str:
    """Return the rightmost LRS3 or VoxCeleb speaker component, else the full clip ID."""
    parts = str(video_id).split("/")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i - 1] == _LRS3_CORPUS and parts[i]:
            return "lrs3_" + parts[i]
    for part in reversed(parts):
        if _VOX_SPEAKER.match(part):
            return part
    return video_id


RESAMPLING_UNITS = ("speaker", "clip")


def cluster_indices(video_ids: np.ndarray, unit: str = "speaker",
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return validated cluster keys and per-clip indices.

    ``unit="clip"`` resamples clips independently, the convention for
    datasets without a speaker key."""
    n = len(video_ids)
    if n == 0:
        raise ValueError("resampling needs at least one clip")
    if unit == "clip":
        return np.asarray(video_ids), np.arange(n)
    if unit != "speaker":
        raise ValueError(f"unknown resampling unit {unit!r}; expected one of "
                         f"{RESAMPLING_UNITS}")
    keys = np.array([speaker_cluster_key(str(v)) for v in video_ids])
    uq, inv, counts = np.unique(keys, return_inverse=True,
                                return_counts=True)
    n_clusters = len(uq)
    if n_clusters == 1:
        raise ValueError(
            f"speaker clustering is degenerate: all {n} clips use the "
            f"single cluster '{uq[0]}'")
    if n_clusters == n:
        raise ValueError(
            f"speaker clustering is degenerate: {n_clusters} clusters "
            f"for {n} clips (every clip is a singleton)")
    singletons = int((counts == 1).sum())
    if singletons > _MAX_SINGLETON_SHARE * n_clusters:
        raise ValueError(
            f"speaker clustering is near-degenerate: {singletons} of "
            f"{n_clusters} clusters are singletons over {n} clips")
    if n_clusters > _MAX_CLUSTERS_PER_CLIP * n:
        raise ValueError(
            f"speaker clustering is near-degenerate: {n_clusters} "
            f"clusters for {n} clips (more than "
            f"{_MAX_CLUSTERS_PER_CLIP:g} clusters per clip)")
    return uq, inv


def multiplicities(inv: np.ndarray, n_clusters: int,
                   draw: np.ndarray) -> np.ndarray:
    """Per-clip multiplicity for one draw of cluster indices (with
    replacement)."""
    return np.bincount(draw, minlength=n_clusters)[inv].astype(np.int64)


# above this share of degenerate replicates, the bootstrap is refused
MAX_DEGENERATE_SHARE = 0.01


@dataclass(frozen=True)
class BootstrapResult:
    n_clusters: int
    n_resamples: int
    seed: int
    unit: str
    # arm -> metric -> (n_resamples,) resampled values
    samples: dict[str, dict[str, np.ndarray]]
    # replicates that drew no ground-truth segment; AP is undefined there
    n_degenerate_resamples: int = 0


def percentile_ci(samples: np.ndarray,
                  confidence_level: float = 0.95) -> tuple[float, float]:
    """Return a percentile interval after dropping degenerate NaN replicates.

    cluster_bootstrap rejects runs with more than 1% degenerate replicates."""

    s = np.asarray(samples, dtype=np.float64)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        raise ValueError(
            f"every one of {s.size} bootstrap replicates is degenerate; "
            f"there is no interval to report")
    a = 100.0 * (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(finite, [a, 100.0 - a])
    return float(lo), float(hi)


def cluster_bootstrap(tables: dict[str, MatchTable], n_resamples: int,
                      seed: int, *, unit: str = "speaker") -> BootstrapResult:
    """Paired cluster bootstrap over arms sharing one clip set: each
    resample draw is applied to every arm, so per-resample differences
    between arms are meaningful. Each draw picks ``n_clusters`` cluster
    indices uniformly with replacement."""
    if n_resamples < 1:
        raise ValueError(
            f"n_resamples={n_resamples}: a bootstrap over zero "
            f"replicates returns empty sample arrays and passes the "
            f"degeneracy guard over nothing")
    names = list(tables)
    if not names:
        raise ValueError("no arms to bootstrap")
    ref = tables[names[0]]
    for n in names[1:]:
        if not np.array_equal(tables[n].video_ids, ref.video_ids):
            raise ValueError(f"clip-order mismatch between arms "
                             f"'{names[0]}' and '{n}'")
        if not np.array_equal(tables[n].n_gt, ref.n_gt):
            # The shared degeneracy guard requires identical GT counts.
            raise ValueError(
                f"arms '{names[0]}' and '{n}' carry different "
                f"per-clip ground-truth counts")
    uq, inv = cluster_indices(ref.video_ids, unit)
    n_clusters = len(uq)
    rng = np.random.default_rng(seed)
    acc: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    n_degenerate = 0
    for _ in range(n_resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        w = multiplicities(inv, n_clusters, draw)
        if int((w * ref.n_gt).sum()) == 0:
            n_degenerate += 1
        for n in names:
            t = tables[n]
            vals = {**{f"ap@{thr:g}": v for thr, v in
                       zip(t.ap_iou_thresholds, ap_from_multiplicities(t, w))},
                    **{f"ar@{k}": v for k, v in
                       zip(t.ar_proposal_counts, ar_from_multiplicities(t, w))}}
            for k, v in vals.items():
                acc[n].setdefault(k, []).append(v)
    if n_degenerate > MAX_DEGENERATE_SHARE * n_resamples:
        raise ValueError(
            f"{n_degenerate} of {n_resamples} bootstrap replicates "
            f"drew no ground-truth segment, exceeding "
            f"{MAX_DEGENERATE_SHARE:.0%}; AP is undefined")
    return BootstrapResult(
        n_clusters=n_clusters, n_resamples=n_resamples, seed=seed, unit=unit,
        samples={n: {k: np.asarray(v) for k, v in d.items()}
                 for n, d in acc.items()},
        n_degenerate_resamples=n_degenerate)
