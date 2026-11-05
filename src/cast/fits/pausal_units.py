"""Fit the maximum interval duration from inter-pausal units."""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..artifacts import FPS
from .words import MIN_CONTENT_WORDS, WordIntervals, reliable_clips

PAUSE_MS: float = 200.0

# Alternative pause thresholds are reported as sensitivity analyses, not as
# candidates for selecting the recorded value.
SENSITIVITY_PAUSE_MS: tuple[float, ...] = (100.0, 150.0, 250.0, 300.0,
                                           500.0)


def segment_pausal_units(words: list[tuple[float, float, str]],
                         pause_s: float,
                         ) -> list[tuple[float, float, int]]:
    """Maximal sub-pause word runs of one clip as (start_s, end_s,
    n_words); the running end is the max over merged words, so
    overlapping alignments cannot open a spurious pause."""
    ipus: list[tuple[float, float, int]] = []
    s0, e0, n = words[0][0], words[0][1], 1
    for s, e, _ in words[1:]:
        if s - e0 < pause_s:
            e0 = max(e0, e)
            n += 1
        else:
            ipus.append((s0, e0, n))
            s0, e0, n = s, e, 1
    ipus.append((s0, e0, n))
    return ipus


def interior_unit_durations(by_clip: WordIntervals, pause_ms: float,
                            min_content_words: int,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Interior-IPU durations (frames) with clip index, clips sorted.
    Interior: neither the first nor the last IPU of its clip."""
    durs: list[float] = []
    cids: list[int] = []
    for ci, vid in enumerate(reliable_clips(by_clip, min_content_words)):
        ipus = segment_pausal_units(by_clip[vid], pause_ms / 1000.0)
        for k, (s, e, _) in enumerate(ipus):
            if 0 < k < len(ipus) - 1:
                durs.append((e - s) * FPS)
                cids.append(ci)
    return np.asarray(durs, float), np.asarray(cids)


@dataclass(frozen=True)
class MaxDurationFit:
    p99_frames: float
    max_duration_frames: int
    n_units: int
    n_clips: int
    p99_ci: tuple[float, float]
    pause_ms: float
    # pause_ms -> (n_units, p99_frames), sensitivity analysis only
    pause_sensitivity: tuple[tuple[float, int, float], ...]
    # the parameters this fit actually ran with
    bootstrap_resamples: int
    bootstrap_seed: int


def fit_max_duration_frames(by_clip: WordIntervals,
                            pause_ms: float = PAUSE_MS,
                            min_content_words: int = MIN_CONTENT_WORDS,
                            bootstrap: int = 2000, seed: int = 0,
                            ) -> MaxDurationFit:
    """Fit the enumeration cap; the clip-clustered p99 CI resamples
    clips with replacement (default_rng(seed), the recorded fit's draw
    sequence)."""
    durs, cids = interior_unit_durations(by_clip, pause_ms,
                                         min_content_words)
    if durs.size == 0:
        raise ValueError("no interior inter-pausal units; cannot fit "
                         "max_duration_frames")
    p99 = float(np.percentile(durs, 99))
    rng = np.random.default_rng(seed)
    uniq = np.unique(cids)
    by = {c: durs[cids == c] for c in uniq}
    vals = []
    for _ in range(bootstrap):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals.append(np.percentile(np.concatenate([by[c] for c in pick]),
                                  99))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    sens = []
    for pm in SENSITIVITY_PAUSE_MS:
        sd, _ = interior_unit_durations(by_clip, pm, min_content_words)
        if sd.size:
            sens.append((pm, int(sd.size), float(np.percentile(sd, 99))))
    return MaxDurationFit(
        p99_frames=p99, max_duration_frames=int(round(p99)),
        n_units=int(durs.size), n_clips=int(uniq.size),
        p99_ci=(float(lo), float(hi)), pause_ms=pause_ms,
        pause_sensitivity=tuple(sens),
        bootstrap_resamples=int(bootstrap), bootstrap_seed=int(seed))
