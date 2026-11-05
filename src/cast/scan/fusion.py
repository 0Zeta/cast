"""Fit and apply candidate, frame, and clip score fusion."""


from __future__ import annotations

import numpy as np

# Precision pool: scan, tier-2 order statistics, and top5_proposal_mean.
TIER2_SUFFIX = ("_low2", "_low5", "_low15", "_top2", "_top5", "_top15")
POOLING_CHANNEL = "top5_proposal_mean"

# Audio-restricted diagnostic variant: group-name prefixes.
AUDIO_PREFIXES = ("cross_modal_agreement_", "video_side_avsr_agreement_",
                  "audio_side_avsr_agreement_",
                  "audio_artifact_score_", "audio_to_video_prediction_",
                  "synchrony_residual_")


def fit_nonneg_logistic(X, y, lam=1.0):
    """Fit nonnegative logistic weights with a free intercept."""
    from scipy.optimize import minimize
    n, k = X.shape
    y = np.asarray(y, dtype=np.float64)

    def obj(params):
        w, b = params[:k], params[k]
        z = X @ w + b
        loss = (float(np.mean(np.logaddexp(0.0, z) - y * z))
                + lam * float(w @ w) / k)
        gz = 1.0 / (1.0 + np.exp(-z)) - y
        gw = X.T @ gz / n + 2.0 * lam * w / k
        gb = float(gz.mean())
        return loss, np.concatenate([gw, [gb]])

    x0 = np.concatenate([np.full(k, 0.1), [0.0]])
    res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * k + [(None, None)])
    return res.x[:k]


def min_fusion_value(rhos):
    """Tippett min-fusion over the channel tails:
    v = -log max(min_c rho_c, 1e-300), NaN-skipping minimum."""
    stack = np.vstack(list(rhos.values()))
    r = np.nanmin(stack, axis=0)
    return -np.log(np.maximum(r, 1e-300))


def candidate_feature_matrix(rhos):
    """(N, n_channels) matrix of -log rho in channel construction
    order, NaN -> 0 (an abstaining channel contributes nothing)."""
    X = np.vstack([-np.log(np.maximum(rhos[k], 1e-300))
                   for k in rhos]).T
    X[~np.isfinite(X)] = 0.0
    return X


def mad_scale(X):
    """Per-clip heteroscedastic channel scaling: each
    candidate channel divided by its within-clip robust scale
    sigma = 1.4826 * MAD, floored at max(0.05 * max_col(sigma), 1e-6);
    abstaining (all-zero) columns stay zero."""
    med = np.median(X, axis=0)
    sig = 1.4826 * np.median(np.abs(X - med), axis=0)
    floor = max(0.05 * float(sig.max()), 1e-6)
    return X / np.maximum(sig, floor)


def nnls_index_sets(feature_names):
    """(nnls_idx, nnls_audio_idx) into the clip feature vector
   ."""
    nnls_idx = np.array([k for k, n in enumerate(feature_names)
                         if n == "scan" or n.endswith(TIER2_SUFFIX)
                         or n == POOLING_CHANNEL])
    nnls_audio_idx = np.array([k for k in nnls_idx
                               if feature_names[k] == "scan"
                               or feature_names[k].startswith(
                                   AUDIO_PREFIXES)])
    return nnls_idx, nnls_audio_idx


def fit_precision_weights(X, log=print, tag=""):
    """Genuine-only NNLS precision pooling: solve
    Sigma w = 1 with w >= 0, Sigma = Cov(X) + ridge
    1e-3 * mean(diag); equal weights as the degenerate-solve
    fallback."""
    from scipy.optimize import nnls as _nnls
    S = np.atleast_2d(np.cov(X.T))
    S = S + 1e-3 * float(np.diag(S).mean()) * np.eye(len(S))
    try:
        w, _ = _nnls(S, np.ones(len(S)))
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("degenerate nnls solution")
    except Exception as e:
        log(f"{tag}nnls fallback to equal weights: {e}")
        w = np.ones(len(S))
    return w


def mix_weights(fusion_w, nnls_w, nnls_idx, n_features):
    """mix_w = 0.5 * (L1-normalized fusion_w + L1-normalized
    zero-expanded nnls_w)."""
    lrn_n = fusion_w / max(np.abs(fusion_w).sum(), 1e-12)
    nnls_full = np.zeros(n_features)
    nnls_full[nnls_idx] = nnls_w
    nnls_n = nnls_full / max(np.abs(nnls_full).sum(), 1e-12)
    return 0.5 * (lrn_n + nnls_n)
