"""Decode non-overlapping proposal intervals."""


from __future__ import annotations

import numpy as np

from .calibration import N_BIN_EDGES

# Fallback insertion rate; configured runs pass ScanConfig.insertion_rate.
INSERTION_RATE = 0.05


def temporal_iou(s1, e1, s2, e2):
    """IoU of [s1, e1] and [s2, e2], inclusive frame indices."""
    inter = np.minimum(e1, e2) - np.maximum(s1, s2) + 1
    union = np.maximum(e1, e2) - np.minimum(s1, s2) + 1
    return np.maximum(inter, 0) / union


def insertion_penalties(n, eps=INSERTION_RATE):
    """beta per candidate from duration-bin multiplicity (label-free);
    the bins are the seven candidate duration bins."""
    nb = np.searchsorted(N_BIN_EDGES, n, side="right") - 1
    beta = np.zeros(len(n))
    for b in np.unique(nb):
        N_b = int((nb == b).sum())
        beta[nb == b] = np.log(max(N_b, 1) / eps)
    return beta


def segmental_decode(start, n, s, T, eps=INSERTION_RATE):
    """Decode MAP segments and marginal inclusion probabilities."""
    N = len(start)
    if N == 0:
        return [], np.zeros(0)
    phi = s - insertion_penalties(n, eps)
    end = start + n - 1                                # inclusive
    by_end = [[] for _ in range(T)]
    by_start = [[] for _ in range(T)]
    for i in range(N):
        by_end[end[i]].append(i)
        by_start[start[i]].append(i)

    # log-space forward: F[t] over frames 1..t (index shifted by 2 so
    # that F[s-2] is addressable for s = 0, 1)
    F = np.full(T + 2, -np.inf)
    F[0] = F[1] = 0.0                                  # F_t = 1 for t <= 0
    for t in range(T):
        terms = [F[t + 1]]                             # frame t real
        for i in by_end[t]:
            terms.append(F[start[i]] + phi[i])         # F[s-2+2] = F[start]
        F[t + 2] = np.logaddexp.reduce(terms)

    G = np.full(T + 2, -np.inf)
    G[T] = G[T + 1] = 0.0                              # G_t = 1 for t > T-1
    for t in range(T - 1, -1, -1):
        terms = [G[t + 1]]
        for i in by_start[t]:
            terms.append(phi[i] + G[min(end[i] + 2, T + 1)])
        G[t] = np.logaddexp.reduce(terms)

    logZ = F[T + 1]
    q = np.exp(F[start] + phi + G[np.minimum(end + 2, T + 1)] - logZ)

    # max-sum MAP with backpointers
    M = np.full(T + 2, -np.inf)
    M[0] = M[1] = 0.0
    back = np.full(T + 2, -1, dtype=np.int64)          # candidate index or -1
    for t in range(T):
        best, arg = M[t + 1], -1
        for i in by_end[t]:
            v = M[start[i]] + phi[i]
            if v > best:
                best, arg = v, i
        M[t + 2] = best
        back[t + 2] = arg
    segs = []
    t = T + 1
    while t > 1:
        i = back[t]
        if i == -1:
            t -= 1
        else:
            segs.append(int(i))
            t = start[i]
    return segs[::-1], q


def decoder_gain(start, n, s, T, eps=INSERTION_RATE):
    """Return the best nonempty gain, or NaN without candidates."""
    N = len(start)
    if N == 0:
        return float("nan")
    phi = s - insertion_penalties(n, eps)
    end = start + n - 1
    by_end = [[] for _ in range(T)]
    for i in range(N):
        by_end[end[i]].append(i)
    M = np.zeros(T + 2)                     # empty prefix scores 0
    M1 = np.full(T + 2, -np.inf)            # forced nonempty
    for t in range(T):
        b = M[t + 1]
        b1 = M1[t + 1]
        for i in by_end[t]:
            v = M[start[i]] + phi[i]
            if v > b:
                b = v
            if v > b1:
                b1 = v
        M[t + 2] = b
        M1[t + 2] = b1
    return float(M1[T + 1])


def nms_proposals(start, n, score, iou_thr=0.5, top_k=50, pre_k=2000):
    """Greedy temporal NMS over candidates: score
    descending, pre-truncated to the top pre_k, non-finite scores
    skipped; returns an index array into the inputs."""
    if len(start) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-score)
    if len(order) > pre_k:
        order = order[:pre_k]
    s = start[order].astype(np.int64)
    e = (start[order] + n[order] - 1).astype(np.int64)
    kept = []
    for i in range(len(order)):
        if not np.isfinite(score[order[i]]):
            continue
        if kept:
            ious = temporal_iou(s[i], e[i], s[kept], e[kept])
            if (ious >= iou_thr).any():
                continue
        kept.append(i)
        if len(kept) >= top_k:
            break
    return order[kept]
