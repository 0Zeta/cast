"""Enumerate and score candidate intervals."""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nulls import MultivariateNullModel, ScalarNullModel

# Step-contrast flank width is clip(ceil(n / 2), 2, 10).
STEP_CONTRAST_FLANK_MIN = 2
STEP_CONTRAST_FLANK_MAX = 10
STEP_CONTRAST_MIN_DURATION = 3


@dataclass(frozen=True)
class ScanGeometry:
    """Interval-scan geometry, from ScanConfig."""
    guard_band_frames: int = 0     # h
    min_interval_frames: int = 25  # m_min
    max_duration_frames: int = 127 # d_max
    edge_margin_frames: int = 8    # n_edge


@dataclass
class CandidateSet:
    """Per-candidate interval statistics of one clip, one score group."""
    start: np.ndarray    # (N,) int32, first frame
    n: np.ndarray        # (N,) int32, duration in frames
    m: np.ndarray        # (N,) int32, safe-complement size
    z0: np.ndarray       # (N,) float64
    zb0: np.ndarray      # (N,) float64, nan where n < edge_margin_frames
    edge: np.ndarray     # (N,) bool
    zt0: np.ndarray = None  # (N,) float64, step contrast

    def __len__(self) -> int:
        return len(self.start)


def edge_weight_cov(n: int):
    """Zero-sum boundary template w and its autocovariances c_w(l)."""
    a = int(np.ceil(n / 4))
    w = np.full(n, -1.0 / (n - 2 * a))
    w[:a] = 1.0 / (2 * a)
    w[n - a:] = 1.0 / (2 * a)
    c = np.correlate(w, w, mode="full")[n - 1:]  # c[l] = sum_t w_t w_{t+l}
    return w, c


def edge_variance_table(rho: np.ndarray, d_max: int,
                        n_edge: int) -> np.ndarray:
    """Upsilon_n for n = 0..d_max; NaN below n_edge."""
    L = len(rho)
    ups = np.full(d_max + 1, np.nan)
    for n in range(n_edge, d_max + 1):
        _, c = edge_weight_cov(n)
        lmax = min(L, n - 1)
        ups[n] = c[0] + 2.0 * float((rho[:lmax] * c[1:lmax + 1]).sum())
    return ups


def step_contrast_flank(n: int) -> int:
    """Per-side flank width of the duration-n step-contrast template."""
    return int(min(max(STEP_CONTRAST_FLANK_MIN, (n + 1) // 2),
                   STEP_CONTRAST_FLANK_MAX))


def _step_contrast_var(rho, n, fL, fR, cache):
    """Return cached inside-minus-flank variance."""
    key = (n, fL, fR)
    v = cache.get(key)
    if v is not None:
        return v
    tot = fL + fR
    w = np.empty(fL + n + fR)
    w[:fL] = -1.0 / tot
    w[fL:fL + n] = 1.0 / n
    w[fL + n:] = -1.0 / tot
    c = np.correlate(w, w, mode="full")[len(w) - 1:]
    lmax = min(len(rho), len(w) - 1)
    v = float(c[0] + 2.0 * (rho[:lmax] * c[1:lmax + 1]).sum())
    v = max(v, 1e-12)
    cache[key] = v
    return v


def scalar_candidate_stats(y, null: ScalarNullModel, geom: ScanGeometry,
                           ups=None, excl=None,
                           pooled_baseline: bool = False) -> CandidateSet:
    """Enumerate candidates of one clip and compute the scalar interval
    statistics."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    h, m_min = geom.guard_band_frames, geom.min_interval_frames
    d_max, n_edge = geom.max_duration_frames, geom.edge_margin_frames
    if ups is None:
        ups = edge_variance_table(null.rho, d_max, n_edge)
    kap = null.kappa
    psi, q = null.psi, null.q

    P = np.zeros(T + 1)
    np.cumsum(y, out=P[1:])
    if excl is None:
        Pm, Km = P, None
    else:
        kept = ~np.asarray(excl, bool)
        Pm = np.zeros(T + 1)
        np.cumsum(y * kept, out=Pm[1:])
        Km = np.zeros(T + 1, dtype=np.int64)
        np.cumsum(kept, out=Km[1:])

    tcache = null.step_contrast_cache
    outs = {k: [] for k in ("start", "n", "m", "z0", "zb0", "edge",
                            "zt0")}
    for d in range(1, min(d_max, T) + 1):
        s = np.arange(0, T - d + 1)
        loL = np.maximum(0, s - h)
        hiR = np.minimum(T, s + d + h)
        if Km is None:
            mL, mR = loL, T - hiR
        else:
            mL, mR = Km[loL], Km[T] - Km[hiR]
        m = mL + mR
        keep = m >= m_min
        if not keep.any():
            continue
        s, loL, hiR, mL, mR, m = (s[keep], loL[keep], hiR[keep],
                                  mL[keep], mR[keep], m[keep])

        SI = P[s + d] - P[s]
        SC = Pm[loL] + (Pm[T] - Pm[hiR])
        # Clamp beyond the fitted table because kappa_n converges with n.
        kmax = len(kap) - 1
        kC = (mL * kap[np.minimum(mL, kmax)]
              + mR * kap[np.minimum(mR, kmax)]) / m
        m_eff = m / kC
        if q and pooled_baseline:
            # no clip effect is estimated; its prior variance stays
            A = psi * np.ones_like(m_eff)
            eta = 0.0
        elif q:
            A = 1.0 / (1.0 / psi + m_eff)
            eta = A * SC / kC
        else:
            A = np.zeros_like(m_eff)
            eta = 0.0
        kn = kap[d]
        n_eff = d / kn
        u = (SI - d * eta) / np.sqrt(d * kn)
        z0 = u / np.sqrt(1.0 + n_eff * A)

        if d >= n_edge:
            a = int(np.ceil(d / 4))
            edges = (P[s + a] - P[s]) + (P[s + d] - P[s + d - a])
            mid = SI - edges
            TI = edges / (2 * a) - mid / (d - 2 * a)
            zb0 = TI / np.sqrt(ups[d])
        else:
            zb0 = np.full(len(s), np.nan)

        if d < STEP_CONTRAST_MIN_DURATION:
            zt0 = np.full(len(s), np.nan)
        else:
            f = step_contrast_flank(d)
            loF = np.maximum(0, s - f)
            hiF = np.minimum(T, s + d + f)
            if Km is None:
                fLk, fRk = s - loF, hiF - (s + d)
                SL = P[s] - P[loF]
                SR = P[hiF] - P[s + d]
            else:
                fLk = Km[s] - Km[loF]
                fRk = Km[hiF] - Km[s + d]
                SL = Pm[s] - Pm[loF]
                SR = Pm[hiF] - Pm[s + d]
            tot = fLk + fRk
            zt0 = np.full(len(s), np.nan)
            ok = tot >= 2
            if ok.any():
                t_c = SI[ok] / d - (SL[ok] + SR[ok]) / tot[ok]
                var = np.empty(int(ok.sum()))
                pair = fLk[ok] * (STEP_CONTRAST_FLANK_MAX + 1) + fRk[ok]
                for pk in np.unique(pair):
                    pfL = int(pk) // (STEP_CONTRAST_FLANK_MAX + 1)
                    pfR = int(pk) % (STEP_CONTRAST_FLANK_MAX + 1)
                    var[pair == pk] = _step_contrast_var(null.rho, d, pfL,
                                                         pfR, tcache)
                zt0[ok] = t_c / np.sqrt(var)

        outs["start"].append(s.astype(np.int32))
        outs["n"].append(np.full(len(s), d, dtype=np.int32))
        outs["m"].append(m.astype(np.int32))
        outs["z0"].append(z0)
        outs["zb0"].append(zb0)
        outs["edge"].append((mL == 0) | (mR == 0))
        outs["zt0"].append(zt0)

    if not outs["start"]:
        e = np.zeros(0)
        return CandidateSet(e.astype(np.int32), e.astype(np.int32),
                            e.astype(np.int32), e, e, e.astype(bool), e)
    return CandidateSet(*[np.concatenate(outs[k]) for k in
                          ("start", "n", "m", "z0", "zb0", "edge",
                           "zt0")])


@dataclass
class MultivariateCandidateSet:
    """Per-candidate standardized statistic vectors of one clip."""
    start: np.ndarray
    n: np.ndarray
    m: np.ndarray
    z0: np.ndarray       # (N, p) float32, low-rank standardized vector
    zb0: np.ndarray      # (N, p) float32, nan where n < edge_margin_frames
    edge: np.ndarray

    def __len__(self) -> int:
        return len(self.start)


def multivariate_candidate_vectors(y, null: MultivariateNullModel,
                                   geom: ScanGeometry, ups=None,
                                   excl=None, pooled_baseline: bool = False,
                                   ) -> MultivariateCandidateSet:
    """Compute standardized multivariate candidate statistics."""
    y = np.asarray(y, dtype=np.float64)
    T, p = y.shape
    h, m_min = geom.guard_band_frames, geom.min_interval_frames
    d_max, n_edge = geom.max_duration_frames, geom.edge_margin_frames
    if ups is None:
        ups = edge_variance_table(null.rho, d_max, n_edge)
    kap = null.kappa
    kmax = len(kap) - 1
    B, psi, q = null.B, null.psi, null.q

    P = np.zeros((T + 1, p))
    np.cumsum(y, axis=0, out=P[1:])
    if excl is None:
        Pm, Km = P, None
    else:
        kept = ~np.asarray(excl, bool)
        Pm = np.zeros((T + 1, p))
        np.cumsum(y * kept[:, None], axis=0, out=Pm[1:])
        Km = np.zeros(T + 1, dtype=np.int64)
        np.cumsum(kept, out=Km[1:])

    outs = {k: [] for k in ("start", "n", "m", "z0", "zb0", "edge")}
    for d in range(1, min(d_max, T) + 1):
        s = np.arange(0, T - d + 1)
        loL = np.maximum(0, s - h)
        hiR = np.minimum(T, s + d + h)
        if Km is None:
            mL, mR = loL, T - hiR
        else:
            mL, mR = Km[loL], Km[T] - Km[hiR]
        m = mL + mR
        keep = m >= m_min
        if not keep.any():
            continue
        s, loL, hiR, mL, mR, m = (s[keep], loL[keep], hiR[keep],
                                  mL[keep], mR[keep], m[keep])

        SI = P[s + d] - P[s]                       # (ns, p)
        SC = Pm[loL] + (Pm[T] - Pm[hiR])
        kC = (mL * kap[np.minimum(mL, kmax)]
              + mR * kap[np.minimum(mR, kmax)]) / m
        m_eff = m / kC
        kn = kap[d]
        n_eff = d / kn

        if q and pooled_baseline:
            A = np.repeat(psi[None, :], len(SI), axis=0)       # (ns, q)
            u = SI / np.sqrt(d * kn)
            v = u @ B                                         # (ns, q)
            z0 = u - (v * (1.0 - 1.0 / np.sqrt(1.0 + n_eff * A))) @ B.T
        elif q:
            A = 1.0 / (1.0 / psi[None, :] + m_eff[:, None])   # (ns, q)
            eta = A * (SC @ B) / kC[:, None]                  # (ns, q)
            u = (SI - d * (eta @ B.T)) / np.sqrt(d * kn)
            v = u @ B                                         # (ns, q)
            z0 = u - (v * (1.0 - 1.0 / np.sqrt(1.0 + n_eff * A))) @ B.T
        else:
            z0 = SI / np.sqrt(d * kn)

        if d >= n_edge:
            a = int(np.ceil(d / 4))
            edges = (P[s + a] - P[s]) + (P[s + d] - P[s + d - a])
            mid = SI - edges
            TI = edges / (2 * a) - mid / (d - 2 * a)
            zb0 = TI / np.sqrt(ups[d])
        else:
            zb0 = np.full((len(s), p), np.nan)

        outs["start"].append(s.astype(np.int32))
        outs["n"].append(np.full(len(s), d, dtype=np.int32))
        outs["m"].append(m.astype(np.int32))
        outs["z0"].append(z0.astype(np.float32))
        outs["zb0"].append(zb0.astype(np.float32))
        outs["edge"].append((mL == 0) | (mR == 0))

    if not outs["start"]:
        e = np.zeros(0)
        z = np.zeros((0, p), np.float32)
        return MultivariateCandidateSet(e.astype(np.int32),
                                        e.astype(np.int32),
                                        e.astype(np.int32), z, z,
                                        e.astype(bool))
    return MultivariateCandidateSet(*[np.concatenate(outs[k]) for k in
                                      ("start", "n", "m", "z0", "zb0",
                                       "edge")])


def stack_prediction_candidates(per_mv: dict, members: tuple
                                ) -> MultivariateCandidateSet:
    """Stack row-aligned multivariate groups on shared geometry."""
    if len(members) == 1:
        return per_mv[members[0]]
    ref = per_mv[members[0]]
    for mname in members[1:]:
        o = per_mv[mname]
        if not (np.array_equal(ref.start, o.start)
                and np.array_equal(ref.n, o.n)
                and np.array_equal(ref.m, o.m)):
            raise ValueError(
                "composite prediction group: candidate geometry misaligned "
                "across the direction pair")
    z0 = np.concatenate([per_mv[mname].z0 for mname in members], axis=1)
    zb0 = np.concatenate([per_mv[mname].zb0 for mname in members], axis=1)
    return MultivariateCandidateSet(ref.start, ref.n, ref.m, z0, zb0,
                                    ref.edge)
