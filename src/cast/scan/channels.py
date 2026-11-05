"""Define candidate, clip, and frame feature order."""


from __future__ import annotations

import numpy as np

from .calibration import CHANNELS, MV_CHANNELS
from .nulls import MultivariateNullModel

# Multivariate prediction pair: the two directions are
# standardized as one composite correction group named "prediction".
PREDICTION_PAIR = ("audio_to_video_prediction", "video_to_audio_prediction")
PREDICTION_COMPOSITE = "prediction"

# Clip-level order statistics: k most extreme frames,
# matched to word-scale segments between the noisy 1-frame min and the
# diluted clip mean.
TOPK = (2, 5, 15)


def correction_groups(mv_groups):
    """Return correction groups in input order, combining the bidirectional prediction pair."""
    if not all(g in mv_groups for g in PREDICTION_PAIR):
        return [(g, (g,)) for g in mv_groups]
    out, placed = [], False
    for g in mv_groups:
        if g in PREDICTION_PAIR:
            if not placed:
                out.append((PREDICTION_COMPOSITE, PREDICTION_PAIR))
                placed = True
        else:
            out.append((g, (g,)))
    return out


def candidate_channel_names(groups, mv_groups=(), channels=CHANNELS,
                            mv_channels=MV_CHANNELS):
    """Candidate channel keys in construction order: scalar groups x
    their channels, then correction groups x their channels."""
    return ([f"{g}:{c}" for g in groups for c in channels]
            + [f"{cg}:{c}" for cg, _ in correction_groups(mv_groups)
               for c in mv_channels])


def clip_stat_names(groups, mv_groups=()):
    """Clip-level statistic names in fusion-vector order:
    scalar groups x (mean, min, low2, low5, low15), then multivariate
    groups x (mean_energy, max_frame_energy, top2, top5, top15)."""
    scalar_stats = ("mean", "min") + tuple(f"low{k}" for k in TOPK)
    mv_stats = (("mean_energy", "max_frame_energy")
                + tuple(f"top{k}" for k in TOPK))
    return ([f"{g}_{stat}" for g in groups for stat in scalar_stats]
            + [f"{g}_{stat}" for g in mv_groups for stat in mv_stats])


def multivariate_clip_stats(y, null: MultivariateNullModel):
    """Clip-level statistics for a whitened (T, p) sequence:
    clip-mean Mahalanobis energy under the factor model (Woodbury) and
    the max raw frame energy."""
    T, p = y.shape
    yb = y.mean(0)
    kap_T = null.kappa[min(T, len(null.kappa) - 1)]
    d = kap_T / T
    if null.q:
        v = null.B.T @ yb
        mean_energy = (yb @ yb
                       - (v * v * null.psi / (null.psi + d)).sum()) / d
    else:
        mean_energy = yb @ yb / d
    frame_energy_max = float((y * y).sum(1).max())
    return float(mean_energy), frame_energy_max


def clip_statistics(y_mat, y_mv, group_cols, nulls, mv_groups, mv_nulls):
    """Compute anomaly-oriented clip statistics."""
    T = len(y_mat)
    out = {}
    for g, col in group_cols.items():
        null = nulls[g]
        y = null.whiten(y_mat[:, col])
        scale = np.sqrt(null.psi
                        + null.kappa[min(T, len(null.kappa) - 1)] / T)
        out[f"{g}_mean"] = -float(y.mean()) / scale
        out[f"{g}_min"] = -float(y.min())
        ys = np.sort(y)
        for k in TOPK:
            out[f"{g}_low{k}"] = -float(ys[:min(k, T)].mean())
    for g in mv_groups:
        null = mv_nulls[g]
        yw = null.whiten(y_mv[g])
        me, fm = multivariate_clip_stats(yw, null)
        out[f"{g}_mean_energy"] = me
        out[f"{g}_max_frame_energy"] = fm
        fe = np.sort((yw * yw).sum(1))[::-1]
        for k in TOPK:
            out[f"{g}_top{k}"] = float(fe[:min(k, T)].mean())
    return out


def frame_feature_names(groups, mv_groups=()):
    """Frame-fusion feature columns: per scalar group the
    whitened drop, per multivariate group the normalized frame
    energy."""
    return ([f"{g}:drop" for g in groups]
            + [f"{g}:energy" for g in mv_groups])


def frame_feature_matrix(y_mat, y_mv, group_cols, nulls, mv_groups,
                         mv_nulls):
    """(T, k) per-frame feature matrix for the frame-level fusion:
    whitened scalar drops (fake = up) + normalized multivariate frame
    energies |y_t|^2 / p of the raw per-direction groups."""
    cols = []
    for g, col in group_cols.items():
        cols.append(-nulls[g].whiten(y_mat[:, col]))
    for g in mv_groups:
        null = mv_nulls[g]
        yw = null.whiten(y_mv[g])
        cols.append((yw * yw).sum(1) / null.p)
    return np.stack(cols, axis=1)
