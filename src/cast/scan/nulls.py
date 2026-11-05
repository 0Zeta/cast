"""Fit scalar and multivariate null models on genuine clips."""


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

# Autocorrelation band and kappa-table conventions.
ACF_BAND = 60             # rho(l) band, lags 1..L
KAPPA_MIN = 0.2           # defensive floor
KAPPA_TABLE_MIN = 750     # kappa is tabulated to max(750, longest clip)
HELDOUT_FRACTION = 0.2    # rank-selection held-out share
FACTOR_RANK_MAX = 4       # multivariate factor-rank sweep 0..min(4, p)
COVARIANCE_SHRINKAGE = 0.05


@dataclass
class ScalarNullModel:
    """Fitted scalar null: whitening plus dependence."""
    mu: float
    sigma_w: float
    psi: float                 # random-effect variance; 0.0 when q == 0
    q: int                     # selected rank (0 or 1)
    rho: np.ndarray            # (L,) pooled autocorrelation at lags 1..L
    kappa: np.ndarray          # (max_n+1,) variance inflation per run length
    L: int = ACF_BAND
    n_fit_clips: int = 0
    # step-contrast template-variance cache, keyed (n, f_L, f_R); one
    # per fitted null, shared across every clip a worker scores
    step_contrast_cache: dict = field(default_factory=dict, repr=False,
                                      compare=False)

    def whiten(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=np.float64) - self.mu) / self.sigma_w


def _clip_mean_loglik(psi, ybar, T):
    v = psi + 1.0 / T
    return -0.5 * np.sum(np.log(2 * np.pi * v) + ybar**2 / v)


def _fit_psi(ybar, T):
    res = minimize_scalar(
        lambda p: -_clip_mean_loglik(p, ybar, T),
        bounds=(1e-8, 100.0), method="bounded",
    )
    return float(res.x)


def kappa_lookup(kappa: np.ndarray, T) -> np.ndarray:
    """kappa_{T} lookup with table clamping; T may be an array."""
    idx = np.clip(np.asarray(T, dtype=int), 2, len(kappa) - 1)
    return kappa[idx]


def fit_scalar_null(seqs, L=ACF_BAND, max_n=KAPPA_TABLE_MIN,
                    kappa_min=KAPPA_MIN, val_frac=HELDOUT_FRACTION,
                    seed=0) -> ScalarNullModel:
    """Fit a scalar null from raw genuine-clip sequences in their supplied order."""
    seqs = [np.asarray(s, dtype=np.float64) for s in seqs if len(s) >= 2]
    if len(seqs) < 10:
        raise ValueError(f"need >=10 genuine clips, got {len(seqs)}")
    clip_means = np.array([s.mean() for s in seqs])
    mu = float(clip_means.mean())

    ss_w = sum(float(((s - s.mean()) ** 2).sum()) for s in seqs)
    dof = sum(len(s) - 1 for s in seqs)
    sigma_w = float(np.sqrt(ss_w / dof))

    ys = [(s - mu) / sigma_w for s in seqs]
    ybar = np.array([y.mean() for y in ys])
    T = np.array([len(y) for y in ys], dtype=np.float64)

    # Centering bias at lag l is ~(T_c-l)/T_c*tau, where
    # tau=1+2*sum(rho); linearity in tau gives a closed-form fixed point.
    num = np.zeros(L)
    bias = np.zeros(L)
    den = 0.0
    for y in ys:
        yc = y - y.mean()
        den += float((yc**2).sum())
        for l in range(1, min(L, len(yc) - 1) + 1):
            num[l - 1] += float((yc[:-l] * yc[l:]).sum())
            bias[l - 1] += (len(yc) - l) / len(yc)
    a = num / max(den, 1e-12)
    b = bias / max(den, 1e-12)
    tau = max((1.0 + 2.0 * float(a.sum()))
              / max(1.0 - 2.0 * float(b.sum()), 0.1), 0.05)
    rho = a + tau * b

    # kappa_n table
    kappa = np.ones(max_n + 1)
    for n in range(2, max_n + 1):
        lmax = min(L, n - 1)
        ls = np.arange(1, lmax + 1)
        kappa[n] = max(kappa_min,
                       1.0 + 2.0 * float(((1.0 - ls / n) * rho[:lmax]).sum()))

    # E[sum(x-xbar)^2]=sigma^2*(T-kappa_T); keep rho/kappa from the
    # provisional whitening and use sum(T-kappa_T) as the final denominator.
    kap_c = kappa_lookup(kappa, T)
    dof_eff = float(np.maximum(T - kap_c, 1.0).sum())
    sigma_w = float(np.sqrt(ss_w / dof_eff))
    ys = [(s - mu) / sigma_w for s in seqs]
    ybar = np.array([y.mean() for y in ys])
    T_eff = T / kap_c

    # Rank selection q in {0,1} on held-out clip-mean likelihood, with
    # the effective count in the mean's variance.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ys))
    n_val = max(1, int(val_frac * len(ys)))
    va, tr = perm[:n_val], perm[n_val:]
    psi_tr = _fit_psi(ybar[tr], T_eff[tr])
    ll_q1 = _clip_mean_loglik(psi_tr, ybar[va], T_eff[va])
    ll_q0 = _clip_mean_loglik(0.0, ybar[va], T_eff[va])
    if ll_q1 > ll_q0:
        q, psi = 1, _fit_psi(ybar, T_eff)
    else:
        q, psi = 0, 0.0

    return ScalarNullModel(mu=mu, sigma_w=sigma_w, psi=psi, q=q, rho=rho,
                           kappa=kappa, L=L, n_fit_clips=len(ys))


@dataclass
class MultivariateNullModel:
    """Fitted multivariate null."""
    mu: np.ndarray        # (p,) clip-balanced mean
    W: np.ndarray         # (p, p) whitening Sigma_w^{-1/2}
    B: np.ndarray         # (p, q) orthonormal columns (q may be 0)
    psi: np.ndarray       # (q,) random-effect variances
    q: int
    rho: np.ndarray       # (L,) trace-averaged autocorrelation, lags 1..L
    kappa: np.ndarray     # (max_n+1,)
    L: int = ACF_BAND
    n_fit_clips: int = 0

    @property
    def p(self) -> int:
        return len(self.mu)

    def whiten(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=np.float64) - self.mu) @ self.W.T


def _whitening(seqs, shrink=COVARIANCE_SHRINKAGE):
    p = seqs[0].shape[1]
    mu = np.mean([s.mean(0) for s in seqs], axis=0)
    S = np.zeros((p, p))
    dof = 0
    for s in seqs:
        c = s - s.mean(0)
        S += c.T @ c
        dof += len(s) - 1
    S /= dof
    S = (1 - shrink) * S + shrink * (np.trace(S) / p) * np.eye(p)
    evals, evecs = np.linalg.eigh(S)
    evals = np.maximum(evals, 1e-8)
    W = evecs @ np.diag(evals**-0.5) @ evecs.T
    return mu, W, dof


def _clip_mean_loglik_mv(B, psi, ybar, T):
    """Sum of log N(ybar_c; 0, B diag(psi) B^T + I/T_c) via Woodbury."""
    p = ybar.shape[1]
    ll = 0.0
    for yb, t in zip(ybar, T):
        d = 1.0 / t
        if len(psi) == 0:
            ll += -0.5 * (p * np.log(2 * np.pi * d) + yb @ yb / d)
            continue
        v = B.T @ yb
        M = np.diag(d / psi) + B.T @ B  # q x q; B orthonormal -> I
        Minv_v = np.linalg.solve(M, v)
        quad = (yb @ yb - v @ Minv_v) / d
        logdet = p * np.log(d) + np.sum(np.log(1.0 + psi / d))
        ll += -0.5 * (p * np.log(2 * np.pi) + logdet + quad)
    return ll


def fit_multivariate_null(seqs, L=ACF_BAND, max_n=KAPPA_TABLE_MIN,
                          kappa_min=KAPPA_MIN, q_max=FACTOR_RANK_MAX,
                          val_frac=HELDOUT_FRACTION, seed=0,
                          shrink=COVARIANCE_SHRINKAGE
                          ) -> MultivariateNullModel:
    """Fit multivariate whitening, autocorrelation, and a held-out-rank clip-mean factor model."""
    seqs = [np.asarray(s, dtype=np.float64) for s in seqs if len(s) >= 2]
    if len(seqs) < 10:
        raise ValueError(f"need >=10 genuine clips, got {len(seqs)}")
    mu, W, dof_naive = _whitening(seqs, shrink)
    ys = [(s - mu) @ W.T for s in seqs]
    ybar = np.stack([y.mean(0) for y in ys])
    T = np.array([len(y) for y in ys], dtype=np.float64)
    p = ybar.shape[1]

    # trace-averaged banded autocorrelation with the tau-proportional
    # clip-centering bias correction (per dim, hence * p)
    num = np.zeros(L)
    bias = np.zeros(L)
    den = 0.0
    for y in ys:
        yc = y - y.mean(0)
        den += float((yc * yc).sum())
        for l in range(1, min(L, len(yc) - 1) + 1):
            num[l - 1] += float((yc[:-l] * yc[l:]).sum())
            bias[l - 1] += p * (len(yc) - l) / len(yc)
    a = num / max(den, 1e-12)
    b = bias / max(den, 1e-12)
    tau = max((1.0 + 2.0 * float(a.sum()))
              / max(1.0 - 2.0 * float(b.sum()), 0.1), 0.05)
    rho = a + tau * b

    kappa = np.ones(max_n + 1)
    for n in range(2, max_n + 1):
        lmax = min(L, n - 1)
        ls = np.arange(1, lmax + 1)
        kappa[n] = max(kappa_min,
                       1.0 + 2.0 * float(((1.0 - ls / n) * rho[:lmax]).sum()))

    # Effective degrees of freedom: S is scale-equivariant through the
    # shrinkage, so W rescales analytically by sqrt(dof_eff/dof).
    kap_c = kappa_lookup(kappa, T)
    dof_eff = float(np.maximum(T - kap_c, 1.0).sum())
    g = float(np.sqrt(dof_eff / dof_naive))
    W = W * g
    ybar = ybar * g
    T_eff = T / kap_c

    # Rank-q factor model on the clip means: balanced-PPCA
    # eigendecomposition. The rank sweeps upward and only replaces the
    # best on a strict likelihood improvement.

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ys))
    n_val = max(1, int(val_frac * len(ys)))
    va, tr = perm[:n_val], perm[n_val:]

    def fit_bq(idx, q):
        Sb = np.cov(ybar[idx].T, bias=True)
        t_h = len(idx) / np.sum(1.0 / T_eff[idx])   # harmonic mean
        evals, evecs = np.linalg.eigh(Sb)
        order = np.argsort(evals)[::-1]
        psi = np.maximum(evals[order[:q]] - 1.0 / t_h, 1e-6)
        return evecs[:, order[:q]], psi

    best = (None, None, 0, -np.inf)
    for q in range(0, min(q_max, p) + 1):
        B, psi = (np.zeros((p, 0)), np.zeros(0)) if q == 0 else fit_bq(tr, q)
        ll = _clip_mean_loglik_mv(B, psi, ybar[va], T_eff[va])
        if ll > best[3]:
            best = (B, psi, q, ll)
    Bv, psiv, q, _ = best
    B, psi = (np.zeros((p, 0)), np.zeros(0)) if q == 0 \
        else fit_bq(np.arange(len(ys)), q)
    # prune numerically-zero factors: a component with psi_j ~ 0 has
    # A_j ~ 0 and no effect on eta or Gamma
    keep = psi > 0.01
    B, psi, q = B[:, keep], psi[keep], int(keep.sum())

    return MultivariateNullModel(mu=mu, W=W, B=B, psi=psi, q=q, rho=rho,
                                 kappa=kappa, L=L, n_fit_clips=len(ys))
