"""Calibrate channel statistics with conformal ranks and Pareto tails."""


from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from scipy.stats import genpareto

from ..config import MULTIVARIATE_CHANNELS, SCALAR_CHANNELS

# Cell partition: 7 duration bins x 3 complement bins x
# edge flag = 42 fine cells; the middle ladder step is the duration bin.
N_BIN_EDGES = np.array([1, 3, 5, 9, 17, 33, 65, 1_000_000])
M_BIN_EDGES = np.array([0, 101, 301, 1_000_000])
N_NBINS = len(N_BIN_EDGES) - 1
N_MBINS = len(M_BIN_EDGES) - 1

# Clip-length bins for every clip-level table.
T_BIN_EDGES = np.array([0, 150, 250, 400, 1_000_000])


def length_bin(T):
    """Return the left-closed clip-length bin containing T."""
    return np.searchsorted(T_BIN_EDGES, T, side="right") - 1

# Admission limits for cells, clip bins, and tail fits.
MIN_CELL_REFERENCES = 100
MIN_CLIP_REFERENCES = 50
MIN_TAIL_EXCEEDANCES = 30
TAIL_SELECTION_FRACTION = 0.1

# Candidate channel sets: scalar groups emit all four,
# multivariate correction groups emit energy and edge only.
CHANNELS = SCALAR_CHANNELS
MV_CHANNELS = MULTIVARIATE_CHANNELS


def _require_finite(x: np.ndarray, where: str) -> None:
    if not np.isfinite(x).all():
        raise FloatingPointError(
            f"non-finite value reached a tail lookup in {where}; "
            f"channels must abstain upstream")


def zhang_stephens_gpd_fit(exc):
    """Fit Zhang-Stephens generalized Pareto parameters."""
    x = np.sort(np.asarray(exc, dtype=np.float64))
    n = len(x)
    m = 20 + int(np.sqrt(n))
    xq = x[max(int(np.round(n / 4.0 + 0.5)) - 1, 0)]
    j = np.arange(1.0, m + 1.0)
    theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (j - 0.5))) / (3.0 * xq)
    theta[theta == 0.0] = 1e-12 / x[-1]
    ll = np.empty(m)
    ks = np.empty(m)
    for i in range(m):
        k = -np.mean(np.log1p(-theta[i] * x))
        ks[i] = k
        ll[i] = n * (np.log(theta[i] / k) + k - 1.0)
    w = 1.0 / np.exp(np.clip(ll[None, :] - ll[:, None], -700, 700)).sum(1)
    theta_hat = float((w * theta).sum())
    if theta_hat == 0.0:
        return 0.0, float(x.mean())
    k_hat = -np.mean(np.log1p(-theta_hat * x))
    return float(-k_hat), float(k_hat / theta_hat)


def cell_ids(n, m, edge):
    """(fine_id, mid_id) per candidate: fine = (n_bin, m_bin, edge),
    mid = n_bin. Left-closed, right-open binning."""
    nb = np.searchsorted(N_BIN_EDGES, n, side="right") - 1
    mb = np.searchsorted(M_BIN_EDGES, m, side="right") - 1
    fine = (nb * N_MBINS + mb) * 2 + edge.astype(np.int64)
    return fine, nb.astype(np.int64)


class TailCalibration:
    """Conformal upper-tail ranks with Pareto smoothing strictly above the exact-rank threshold."""

    def __init__(self, values, zeta=TAIL_SELECTION_FRACTION,
                 min_exc=MIN_TAIL_EXCEEDANCES):
        v = np.sort(np.asarray(values, dtype=np.float64))
        if len(v) == 0:
            raise ValueError("TailCalibration requires at least one reference value")
        self.sorted = v
        self.N = len(v)
        self.gpd = None
        if self.N >= 4 * min_exc:
            n_t = max(min_exc, int(np.ceil(zeta * self.N)))
            u = v[self.N - n_t - 1]
            exc = v[v > u] - u
            if len(exc) >= min_exc:
                xi, sigma = zhang_stephens_gpd_fit(exc)
                # Bound xi to preserve ordering: negative xi has a finite
                # endpoint; xi above one produces an unstable heavy tail.

                if not (np.isfinite(xi) and np.isfinite(sigma)
                        and sigma > 0):
                    xi, sigma = 0.0, float(exc.mean())
                elif xi < 0.0:
                    xi, sigma = 0.0, float(exc.mean())
                xi = min(xi, 1.0)
                self.gpd = (float(u), float(xi), float(sigma),
                            float(len(exc) / self.N))

    def sf(self, x):
        x = np.asarray(x, dtype=np.float64)
        _require_finite(x, "TailCalibration.sf")
        n_ge = self.N - np.searchsorted(self.sorted, x, side="left")
        p = (1.0 + n_ge) / (self.N + 1.0)
        if self.gpd is not None:
            u, xi, sigma, zeta_hat = self.gpd
            hi = x > u
            if hi.any():
                p_tail = zeta_hat * genpareto.sf(x[hi] - u, xi, loc=0.0,
                                                 scale=sigma)
                p[hi] = np.maximum(p_tail, 1e-300)
        return p

    def sf_exact(self, x):
        """Pure conformal rank p = (1 + #{ref >= x})/(N + 1), floor
        1/(N + 1); no tail smoothing."""
        x = np.asarray(x, dtype=np.float64)
        _require_finite(x, "TailCalibration.sf_exact")
        n_ge = self.N - np.searchsorted(self.sorted, x, side="left")
        return (1.0 + n_ge) / (self.N + 1.0)


class CellCalibration:
    """Coarseness ladder of TailCalibrations: fine cell -> duration bin
    -> global."""

    def __init__(self, fine, mid, values, zeta=TAIL_SELECTION_FRACTION,
                 min_ref=MIN_CELL_REFERENCES,
                 min_exc=MIN_TAIL_EXCEEDANCES):
        self.min_ref = min_ref
        self.fine_tables = {}
        self.mid_tables = {}
        for cid in np.unique(fine):
            sel = values[fine == cid]
            if len(sel) >= min_ref:
                self.fine_tables[int(cid)] = TailCalibration(sel, zeta,
                                                             min_exc)
        for cid in np.unique(mid):
            sel = values[mid == cid]
            if len(sel) >= min_ref:
                self.mid_tables[int(cid)] = TailCalibration(sel, zeta,
                                                            min_exc)
        self.global_table = TailCalibration(values, zeta, min_exc)

    def sf(self, fine, mid, values):
        out = np.empty(len(values))
        done = np.zeros(len(values), dtype=bool)
        for cid in np.unique(fine):
            t = self.fine_tables.get(int(cid))
            if t is None:
                continue
            sel = fine == cid
            out[sel] = t.sf(values[sel])
            done[sel] = True
        rest = ~done
        if rest.any():
            for cid in np.unique(mid[rest]):
                t = self.mid_tables.get(int(cid))
                sel = rest & (mid == cid)
                if t is None:
                    out[sel] = self.global_table.sf(values[sel])
                else:
                    out[sel] = t.sf(values[sel])
        return out


class ClipLengthCalibration:
    """Clip-length upper-tail tables with a global fallback."""

    def __init__(self, T, values, min_ref=MIN_CLIP_REFERENCES,
                 min_exc=MIN_TAIL_EXCEEDANCES):
        T = np.asarray(T)
        values = np.asarray(values, dtype=np.float64)
        if len(T) != len(values):
            raise ValueError(
                f"ClipLengthCalibration: {len(T)} clip lengths against "
                f"{len(values)} reference values")
        keep = np.isfinite(values)
        T, values = T[keep], values[keep]
        if len(values) == 0:
            raise ValueError(
                f"ClipLengthCalibration needs at least one finite reference; "
                f"got 0 of {len(keep)}")
        bins = length_bin(T)
        self.tables = {}
        n_bins = len(T_BIN_EDGES) - 1
        self.bin_counts = {b: int(np.sum(bins == b))
                           for b in range(n_bins)}
        for b in np.unique(bins):
            vals = values[bins == b]
            if len(vals) >= min_ref:
                self.tables[int(b)] = TailCalibration(vals, min_exc=min_exc)
        self.fallback_bins = sorted(set(range(n_bins)) - set(self.tables))
        if self.fallback_bins:
            # Sparse bins fall back to marginal calibration; warn because
            # this removes length conditioning.
            edges = [f"[{T_BIN_EDGES[b]}, {T_BIN_EDGES[b + 1]})"
                     for b in self.fallback_bins]
            counts = [self.bin_counts[b] for b in self.fallback_bins]
            warnings.warn(
                f"ClipLengthCalibration: length bin(s) {edges} hold "
                f"{counts} reference(s), fewer than min_ref={min_ref}; "
                f"clips in these bins fall back to the GLOBAL table, "
                f"which weakens length-conditional calibration there",
                stacklevel=2)
        self.global_table = TailCalibration(values, min_exc=min_exc)

    def sf(self, T, x):
        b = int(length_bin(T))
        return float(self.tables.get(b, self.global_table)
                     .sf(np.array([x]))[0])

    def sf_exact(self, T, x):
        b = int(length_bin(T))
        return float(self.tables.get(b, self.global_table)
                     .sf_exact(np.array([x]))[0])


@dataclass
class MomentCorrection:
    """Global affine corrections for raw scalar statistics."""
    mu_z: float
    sd_z: float
    mu_zb: float
    sd_zb: float
    mu_zt: float = 0.0
    sd_zt: float = 1.0

    @classmethod
    def fit(cls, z0, zb0, zt0=None):
        """z0 is not NaN-filtered; zb0/zt0 are."""
        zb = zb0[np.isfinite(zb0)]
        zt = (zt0[np.isfinite(zt0)] if zt0 is not None
              else np.zeros(0))
        return cls(
            mu_z=float(np.mean(z0)), sd_z=float(max(np.std(z0), 1e-6)),
            mu_zb=float(np.mean(zb)) if len(zb) else 0.0,
            sd_zb=float(max(np.std(zb), 1e-6)) if len(zb) else 1.0,
            mu_zt=float(np.mean(zt)) if len(zt) else 0.0,
            sd_zt=float(max(np.std(zt), 1e-6)) if len(zt) else 1.0,
        )

    def apply(self, cands):
        z = (cands.z0 - self.mu_z) / self.sd_z
        zb = (cands.zb0 - self.mu_zb) / self.sd_zb
        zt = ((cands.zt0 - self.mu_zt) / self.sd_zt
              if cands.zt0 is not None
              else np.full(len(z), np.nan))
        return z, zb, zt


@dataclass
class DirectionalCorrection:
    """Pooled directional first/second-moment correction for a
    multivariate correction group."""
    mu_z: np.ndarray     # (p,)
    Wc_z: np.ndarray     # (p, p) = C^{-1/2}
    mu_zb: np.ndarray
    Wc_zb: np.ndarray

    @staticmethod
    def _fit_one(Z, shrink=0.05):
        mu = Z.mean(0)
        C = np.cov((Z - mu).T)
        p = len(mu)
        C = (1 - shrink) * C + shrink * (np.trace(C) / p) * np.eye(p)
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, 1e-8)
        return mu, evecs @ np.diag(evals**-0.5) @ evecs.T

    @classmethod
    def fit(cls, z0, zb0):
        mu_z, Wc_z = cls._fit_one(np.asarray(z0, np.float64))
        fb = np.isfinite(zb0).all(axis=1)
        if fb.sum() >= 10 * zb0.shape[1]:
            mu_zb, Wc_zb = cls._fit_one(np.asarray(zb0[fb], np.float64))
        else:
            # Use (0, I) when fewer than 10 fully finite rows exist.
            mu_zb = np.zeros(zb0.shape[1])
            Wc_zb = np.eye(zb0.shape[1])
        return cls(mu_z, Wc_z, mu_zb, Wc_zb)

    def energies(self, cands):
        """Corrected energies ||C^{-1/2}(z0 - mu)||^2 per channel."""
        z = (cands.z0 - self.mu_z) @ self.Wc_z.T
        e = (z * z).sum(1)
        zb = (cands.zb0 - self.mu_zb) @ self.Wc_zb.T
        eb = (zb * zb).sum(1)
        return e, eb


def channel_values(z, zb, zt=None):
    """Build channel values from corrected statistics."""

    if zt is None:
        zt = np.full(len(np.atleast_1d(z)), np.nan)
    return {"energy": z * z, "drop": -z, "edge": zb * zb,
            "step_contrast": -zt}


def sample_one_per_cell(fine, rng):
    """Sample one candidate per nonempty cell."""
    idx = rng.permutation(len(fine))
    _, first = np.unique(fine[idx], return_index=True)
    return np.sort(idx[first])
