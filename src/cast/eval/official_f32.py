"""Reproduce the compiled evaluator's float32 reductions."""


from __future__ import annotations

import numpy as np


ARITHMETIC_IDENTITY = "official_f32_unrolled_v1"
_F32_EXACT_INT_LIMIT = 1 << 24


def _unrolled_sparse_sum(contributions: np.ndarray,
                         positions: np.ndarray,
                         full_length: int) -> np.float32:
    """Reproduce NumPy's eight-lane float32 fold over descending nonzero positions."""
    values = np.asarray(contributions, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.int64)
    if len(values) != len(positions):
        raise ValueError((len(values), len(positions)))
    if len(positions) and not np.all(positions[:-1] > positions[1:]):
        raise ValueError("sparse positions must be strictly descending")

    main_length = (int(full_length) // 8) * 8
    lanes = np.zeros(8, dtype=np.float32)
    in_main = positions < main_length
    for lane in range(8):
        # Descending sparse positions become the fold's ascending traversal.
        lane_values = values[in_main & ((positions % 8) == lane)][::-1]
        if len(lane_values):
            lanes[lane] = np.cumsum(lane_values, dtype=np.float32)[-1]

    total = np.float32(0)
    for left, right in ((0, 4), (1, 5), (2, 6), (3, 7)):
        pair = np.float32(lanes[left] + lanes[right])
        total = np.float32(total + pair)

    # ndarray handles the zero-to-seven tail after combining the lanes.
    for value in values[~in_main][::-1]:
        total = np.float32(total + value)
    return total


def official_ap_from_tp(is_tp: np.ndarray, n_labels: int) -> float:
    """Official all-point AP for an already score-sorted TP vector."""
    tp = np.asarray(is_tp, dtype=bool)
    n_labels = int(n_labels)
    if n_labels <= 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    if len(tp) >= _F32_EXACT_INT_LIMIT or n_labels >= _F32_EXACT_INT_LIMIT:
        raise ValueError("official f32 counter exceeds its exact-integer range")

    count = np.cumsum(tp, dtype=np.float32)
    denominator = np.arange(1, len(tp) + 1, dtype=np.float32)
    precision = count / denominator
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall = count / np.float32(n_labels)

    # The reference drops rank 1; TP index k maps to lane (len(tp)-k) % 8.
    tp_index = np.flatnonzero(tp)
    tp_index = tp_index[tp_index >= 1]
    positions = len(tp) - tp_index
    recall_step = recall[tp_index] - recall[tp_index - 1]
    contributions = np.asarray(
        recall_step * envelope[tp_index], dtype=np.float32)
    return float(_unrolled_sparse_sum(contributions, positions, len(tp)))


def official_ar_mean(tp_counts: np.ndarray, n_labels: int) -> float:
    """Official sequential-f32 AR mean for one proposal budget."""
    counts = np.asarray(tp_counts)
    n_labels = int(n_labels)
    if n_labels <= 0:
        return float("nan")
    if n_labels >= _F32_EXACT_INT_LIMIT:
        raise ValueError("official f32 counter exceeds its exact-integer range")
    if len(counts) == 0:
        raise ValueError("official AR needs a nonempty IoU grid")
    recall = counts.astype(np.float32) / np.float32(n_labels)
    total = np.float32(0)
    for value in recall:
        total = np.float32(total + value)
    return float(np.float32(total / np.float32(len(recall))))


__all__ = ["ARITHMETIC_IDENTITY", "official_ap_from_tp",
           "official_ar_mean"]
