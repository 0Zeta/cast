"""Fit the self-splice duration law from content-word lengths."""


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .words import (MIN_CONTENT_WORDS, WordIntervals,
                    content_word_duration_frames, reliable_clips)


def log_moments(durations: np.ndarray) -> tuple[float, float]:
    """(mean, ddof-0 std) of log durations; ddof 0 because the corpus
    generator's sigma is the ML scale."""
    lg = np.log(durations)
    return float(lg.mean()), float(lg.std())


def _percentile_ci(values: list[float]) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def clustered_durations(by_clip: WordIntervals, min_content_words: int,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Content-word durations (frames) with clip index, clips sorted."""
    durs: list[float] = []
    cids: list[int] = []
    for ci, vid in enumerate(reliable_clips(by_clip, min_content_words)):
        for d in content_word_duration_frames(by_clip[vid]):
            durs.append(d)
            cids.append(ci)
    return np.asarray(durs, float), np.asarray(cids)


@dataclass(frozen=True)
class DurationLawFit:
    mu: float
    sigma: float
    p99_frames: float
    cap_frames: int
    n_clips: int
    n_words: int
    mu_ci: tuple[float, float]
    sigma_ci: tuple[float, float]
    p99_ci: tuple[float, float]
    # the parameters this fit actually ran with, so a record cannot
    # describe a resample count or a seed the fit never used
    bootstrap_resamples: int
    bootstrap_seed: int

    @property
    def fit_quotation_4dp(self) -> dict[str, float | int]:
        """The fitted estimates, rounded to four decimals for quoting.

        These summarise the fit. The generator used different literals; see
        cast.corpus.DEPLOYED_PLACEMENT_LAW.
        """

        return {"mu": round(self.mu, 4), "sigma": round(self.sigma, 4),
                "max_frames": self.cap_frames}


def fit_duration_law(by_clip: WordIntervals,
                     min_content_words: int = MIN_CONTENT_WORDS,
                     bootstrap: int = 2000, seed: int = 0,
                     ) -> DurationLawFit:
    """Fit the duration law from loaded word alignments.

    The clip-clustered bootstrap resamples clips with replacement
    (default_rng(seed), the recorded fit's draw sequence).
    """
    durs, cids = clustered_durations(by_clip, min_content_words)
    if durs.size == 0:
        raise ValueError("no content-word tokens after the reliability "
                         "filter; cannot fit the duration law")
    mu, sigma = log_moments(durs)
    p99 = float(np.percentile(durs, 99))
    rng = np.random.default_rng(seed)
    uniq = np.unique(cids)
    mus, sigmas, p99s = [], [], []
    for _ in range(bootstrap):
        pick = rng.choice(uniq, uniq.size, replace=True)
        sel = np.concatenate([durs[cids == c] for c in pick])
        m, s = log_moments(sel)
        mus.append(m)
        sigmas.append(s)
        p99s.append(float(np.percentile(sel, 99)))
    return DurationLawFit(
        mu=mu, sigma=sigma, p99_frames=p99, cap_frames=math.floor(p99),
        n_clips=int(uniq.size), n_words=int(durs.size),
        mu_ci=_percentile_ci(mus), sigma_ci=_percentile_ci(sigmas),
        p99_ci=_percentile_ci(p99s),
        bootstrap_resamples=int(bootstrap), bootstrap_seed=int(seed))
