"""Compute interval IoU and reference-compatible true-positive assignments."""


from __future__ import annotations

import numpy as np


def iou_matrix(pred_start_s: np.ndarray, pred_end_s: np.ndarray,
               gt_start_s: np.ndarray, gt_end_s: np.ndarray) -> np.ndarray:
    """Return float32 pairwise interval IoU with shape (n_pred, n_gt).

    Zero-width coincidence yields NaN and therefore no strict-threshold match."""
    ps = np.asarray(pred_start_s, dtype=np.float32)[:, None]
    pe = np.asarray(pred_end_s, dtype=np.float32)[:, None]
    gs = np.asarray(gt_start_s, dtype=np.float32)[None, :]
    ge = np.asarray(gt_end_s, dtype=np.float32)[None, :]
    inter = np.maximum(np.minimum(pe, ge) - np.maximum(ps, gs),
                       np.float32(0.0))
    union = np.maximum(pe, ge) - np.minimum(ps, gs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (inter / union).astype(np.float32)


def greedy_assign(iou: np.ndarray, threshold: float) -> np.ndarray:
    """Greedily claim score-ordered proposals at a float32 IoU threshold."""

    # threshold compared in float32, like the reference
    pot = iou > np.float32(threshold)
    n_props, n_gt = iou.shape
    claimed = np.zeros(n_props, dtype=bool)
    for g in range(n_gt):
        for p in np.flatnonzero(pot[:, g]):
            if not claimed[p]:
                claimed[p] = True
                break
    return claimed
