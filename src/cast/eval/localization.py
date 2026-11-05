"""Compute AV-Deepfake1M AP and AR metrics in NumPy."""


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..artifacts import FPS, LabelTable, ProposalTable
from .matching import greedy_assign, iou_matrix
from .official_f32 import (ARITHMETIC_IDENTITY, official_ap_from_tp,
                           official_ar_mean)

AP_IOU_THRESHOLDS: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
AR_PROPOSAL_COUNTS: tuple[int, ...] = (50, 30, 20, 10, 5, 2, 1)
AR_IOU_GRID: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7,
                                  0.75, 0.8, 0.85, 0.9, 0.95)


@dataclass(frozen=True)
class EvaluatorIdentity:
    """A content-independent declaration of the metric grids."""

    name: str
    ap_iou_thresholds: tuple[float, ...]
    ar_proposal_counts: tuple[int, ...]
    ar_iou_grid: tuple[float, ...]

    def __post_init__(self) -> None:
        for label, values in (
                ("AP thresholds", self.ap_iou_thresholds),
                ("AR proposal counts", self.ar_proposal_counts),
                ("AR IoU grid", self.ar_iou_grid)):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{label} must be nonempty and unique")

    def to_dict(self) -> dict:
        return {"name": self.name, "arithmetic": ARITHMETIC_IDENTITY,
                "ap_iou_thresholds": list(self.ap_iou_thresholds),
                "ar_proposal_counts": list(self.ar_proposal_counts),
                "ar_iou_grid": list(self.ar_iou_grid)}


OFFICIAL_EVALUATOR = EvaluatorIdentity(
    name="avdeepfake1m_official_f32_v1",
    ap_iou_thresholds=AP_IOU_THRESHOLDS,
    ar_proposal_counts=AR_PROPOSAL_COUNTS,
    ar_iou_grid=AR_IOU_GRID)


def _multiplicities(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights)
    if w.ndim != 1 or not np.issubdtype(w.dtype, np.number):
        raise ValueError("clip multiplicities must be a numeric vector")
    if not np.isfinite(w).all() or (w < 0).any() \
            or not np.equal(w, np.floor(w)).all():
        raise ValueError(
            "clip multiplicities must be finite nonnegative integers")
    return w.astype(np.int64, copy=False)

# reference placeholder score for clips with no proposals; sits below
# every real proposal in the pooled ranking
PLACEHOLDER_SCORE = np.float32(-1e30)


def proposal_seconds(video_ids: np.ndarray, start_frame: np.ndarray,
                     end_frame: np.ndarray,
                     clip_frame_rate: Mapping[str, float] | None,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Float32 proposal spans in seconds: frame f covers [f, f + 1) / rate.

    ``clip_frame_rate`` maps every clip to its own frame rate; without it the
    constant 25-fps convention applies."""
    if clip_frame_rate is None:
        rate = np.full(len(video_ids), FPS)
    else:
        missing = sorted(set(video_ids.tolist()) - set(clip_frame_rate))
        if missing:
            raise ValueError(f"{len(missing)} proposal clips have no frame "
                             f"rate (e.g. {missing[:3]})")
        rate = np.array([clip_frame_rate[v] for v in video_ids],
                        dtype=np.float64)
        if not (np.isfinite(rate).all() and (rate > 0).all()):
            raise ValueError("clip frame rates must be finite and positive")
    return ((start_frame / rate).astype(np.float32),
            ((end_frame + 1) / rate).astype(np.float32))


@dataclass(frozen=True)
class MatchTable:
    """Resample-invariant per-clip matching results for one proposal
    set against one label set."""

    video_ids: np.ndarray        # (C,)
    n_gt: np.ndarray             # (C,) ground-truth segments per clip
    ar_hits: np.ndarray          # (C, n_grid, n_k) gt hit counts for AR
    clip_index_ranked: np.ndarray  # (P,) clip of each pooled-rank row
    is_tp_ranked: np.ndarray     # (n_thr, P) TP flags in pooled order
    ap_iou_thresholds: tuple[float, ...]
    ar_proposal_counts: tuple[int, ...]
    ar_iou_grid: tuple[float, ...]


def match_proposals(proposals: ProposalTable, labels: LabelTable,
                    ap_iou_thresholds: tuple[float, ...] = AP_IOU_THRESHOLDS,
                    ar_proposal_counts: tuple[int, ...] = AR_PROPOSAL_COUNTS,
                    ar_iou_grid: tuple[float, ...] = AR_IOU_GRID,
                    clip_frame_rate: Mapping[str, float] | None = None,
                    ) -> MatchTable:
    """Match a proposal table against ground truth. Every proposal must
    belong to a labelled clip; unknown video_ids raise instead of being
    dropped."""
    unknown = set(proposals.video_id) - set(labels.video_id)
    if unknown:
        raise ValueError(
            f"{len(unknown)} proposal video_ids are not in the label "
            f"table (e.g. {sorted(unknown)[:3]}); subset them explicitly")

    # per-clip descending-score blocks, ties kept in file order
    # (reference wrapper convention)
    df = proposals.to_dataframe().sort_values(
        ["video_id", "score"], ascending=[True, False], kind="stable")
    vids_p = df.video_id.to_numpy()
    start_s, end_s = proposal_seconds(vids_p, df.start_frame.to_numpy(),
                                      df.end_frame.to_numpy(),
                                      clip_frame_rate)
    score = df.score.to_numpy().astype(np.float32)
    bounds: dict[str, tuple[int, int]] = {}
    if len(vids_p):
        uq, first = np.unique(vids_p, return_index=True)
        last = np.r_[first[1:], len(vids_p)]
        bounds = {v: (int(a), int(b)) for v, a, b in zip(uq, first, last)}

    n_clips = len(labels.video_id)
    thr32 = [np.float32(t) for t in ap_iou_thresholds]
    grid32 = np.array(ar_iou_grid, dtype=np.float32)
    n_gt = np.zeros(n_clips, dtype=np.int64)
    ar_hits = np.zeros((n_clips, len(ar_iou_grid), len(ar_proposal_counts)),
                       dtype=np.int32)
    conf_l, clip_l = [], []
    istp_l: list[list[np.ndarray]] = [[] for _ in ap_iou_thresholds]

    for ci, vid in enumerate(labels.video_id):
        b = bounds.get(vid)
        if b is None:
            ps = np.zeros(1, dtype=np.float32)
            pe = np.zeros(1, dtype=np.float32)
            sc = np.array([PLACEHOLDER_SCORE], dtype=np.float32)
        else:
            a, z = b
            ps, pe, sc = start_s[a:z], end_s[a:z], score[a:z]
        n_props = len(sc)
        conf_l.append(sc)
        clip_l.append(np.full(n_props, ci, dtype=np.int32))
        segs = labels.segments[ci]
        n_gt[ci] = len(segs)
        if len(segs) == 0:
            for t in range(len(ap_iou_thresholds)):
                istp_l[t].append(np.zeros(n_props, dtype=bool))
            continue
        ious = iou_matrix(ps, pe, segs[:, 0], segs[:, 1])
        for t, thr in enumerate(thr32):
            istp_l[t].append(greedy_assign(ious, thr))
        running_max = np.maximum.accumulate(ious, axis=0)
        for ki, k in enumerate(ar_proposal_counts):
            row = running_max[min(k, n_props) - 1]
            ar_hits[ci, :, ki] = (row[None, :] > grid32[:, None]).sum(1)

    conf = np.concatenate(conf_l)
    clip_index = np.concatenate(clip_l)
    istp = np.stack([np.concatenate(x) for x in istp_l])

    order = np.argsort(-conf, kind="stable")   # pooled stable desc sort
    return MatchTable(video_ids=np.asarray(labels.video_id),
                      n_gt=n_gt, ar_hits=ar_hits,
                      clip_index_ranked=clip_index[order].astype(np.int32),
                      is_tp_ranked=istp[:, order],
                      ap_iou_thresholds=tuple(ap_iou_thresholds),
                      ar_proposal_counts=tuple(ar_proposal_counts),
                      ar_iou_grid=tuple(ar_iou_grid))


def ap_from_multiplicities(table: MatchTable, w: np.ndarray) -> list[float]:
    """Official AP at each IoU threshold with per-clip multiplicities w."""
    w = _multiplicities(w)
    n_gt = float(np.dot(table.n_gt, w))
    if n_gt <= 0:
        return [float("nan")] * len(table.ap_iou_thresholds)
    rep = w[table.clip_index_ranked]
    return [official_ap_from_tp(np.repeat(table.is_tp_ranked[t], rep), int(n_gt))
            for t in range(len(table.ap_iou_thresholds))]


def ar_from_multiplicities(table: MatchTable, w: np.ndarray) -> list[float]:
    """Official AR@k (averaged over the IoU grid) with per-clip multiplicities w."""
    w = _multiplicities(w)
    n_gt = float(np.dot(table.n_gt, w))
    if n_gt <= 0:
        return [float("nan")] * len(table.ar_proposal_counts)
    hits = np.tensordot(w, table.ar_hits, axes=(0, 0))   # (n_grid, n_k)
    return [official_ar_mean(hits[:, ki], int(n_gt))
            for ki in range(len(table.ar_proposal_counts))]


def subset_tables(proposals: ProposalTable, labels: LabelTable,
                  video_ids) -> tuple[ProposalTable, LabelTable]:
    """Subset labels and proposals together."""
    keep = set(np.asarray(video_ids, dtype=object).tolist())
    unknown = sorted(keep - set(labels.video_id.tolist()))
    if unknown:
        raise ValueError(
            f"{len(unknown)} requested clips are not in the label "
            f"table (e.g. {unknown[:3]})")
    idx = [i for i, v in enumerate(labels.video_id) if v in keep]
    sub_labels = LabelTable(
        video_id=labels.video_id[idx],
        segments=tuple(labels.segments[i] for i in idx))
    mask = np.array([v in keep for v in proposals.video_id], dtype=bool)
    sub_proposals = ProposalTable(
        video_id=proposals.video_id[mask],
        start_frame=proposals.start_frame[mask],
        end_frame=proposals.end_frame[mask],
        score=proposals.score[mask])
    return sub_proposals, sub_labels


def evaluate_per_type(proposals: ProposalTable, labels: LabelTable,
                      video_ids,
                      ap_iou_thresholds: tuple[float, ...] = AP_IOU_THRESHOLDS,
                      ar_proposal_counts: tuple[int, ...] = AR_PROPOSAL_COUNTS,
                      ar_iou_grid: tuple[float, ...] = AR_IOU_GRID,
                      clip_frame_rate: Mapping[str, float] | None = None,
                      ) -> dict[str, float]:
    """Point estimates on a clip subset (all genuine clips plus one
    manipulation subtype). The subset
    is applied through subset_tables, which reads the labels."""
    sub_p, sub_l = subset_tables(proposals, labels, video_ids)
    return point_estimates(match_proposals(
        sub_p, sub_l, ap_iou_thresholds=ap_iou_thresholds,
        ar_proposal_counts=ar_proposal_counts,
        ar_iou_grid=ar_iou_grid, clip_frame_rate=clip_frame_rate))


def point_estimates(table: MatchTable) -> dict[str, float]:
    """Unresampled official point estimates, keys ap@<iou> and ar@<k>."""
    ones = np.ones(len(table.video_ids), dtype=np.int64)
    ap = ap_from_multiplicities(table, ones)
    ar = ar_from_multiplicities(table, ones)
    return {**{f"ap@{t:g}": v
               for t, v in zip(table.ap_iou_thresholds, ap)},
            **{f"ar@{k}": v
               for k, v in zip(table.ar_proposal_counts, ar)}}


def evaluate_with_identity(
        proposals: ProposalTable, labels: LabelTable,
        evaluator: EvaluatorIdentity,
        clip_frame_rate: Mapping[str, float] | None = None,
        ) -> dict:
    """Evaluate an explicit metric profile and tag the returned metrics with its identity."""

    if not isinstance(evaluator, EvaluatorIdentity):
        raise TypeError("evaluator must be an EvaluatorIdentity")
    table = match_proposals(
        proposals, labels,
        ap_iou_thresholds=evaluator.ap_iou_thresholds,
        ar_proposal_counts=evaluator.ar_proposal_counts,
        ar_iou_grid=evaluator.ar_iou_grid, clip_frame_rate=clip_frame_rate)
    metrics = point_estimates(table)
    expected = (len(evaluator.ap_iou_thresholds)
                + len(evaluator.ar_proposal_counts))
    if len(metrics) != expected:
        raise AssertionError(
            f"evaluator {evaluator.name!r} emitted {len(metrics)} metrics, "
            f"expected {expected}")
    return {"evaluator": evaluator.to_dict(), "metrics": metrics}
