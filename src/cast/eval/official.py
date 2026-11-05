"""Invoke the compiled AV-Deepfake1M evaluator."""


from __future__ import annotations

import importlib.util
import json
import tempfile

import numpy as np
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from ..artifacts import LabelTable, ProposalTable
from .localization import (AP_IOU_THRESHOLDS, AR_IOU_GRID,
                           AR_PROPOSAL_COUNTS, proposal_seconds)


def load_evaluator() -> ModuleType:
    """Load the wheel's compiled `_evaluation` extension directly;
    importing the `avdeepfake1m` package itself would pull in its full
    training dependencies."""
    spec = importlib.util.find_spec("avdeepfake1m")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "the official AV-Deepfake1M evaluator wheel is not "
            "installed; install it with: pip install --no-deps "
            "avdeepfake1m==0.0.4")
    so = [p for d in spec.submodule_search_locations
          for p in sorted(Path(d).glob("_evaluation*.so"))]
    if not so:
        raise RuntimeError(
            "avdeepfake1m is installed but its compiled _evaluation "
            "extension was not found next to the package")
    mspec = importlib.util.spec_from_file_location(
        "avdeepfake1m._evaluation", so[0])
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    return mod


def evaluate_official(proposals: ProposalTable, labels: LabelTable,
                      ap_iou_thresholds: tuple[float, ...] =
                      AP_IOU_THRESHOLDS,
                      ar_proposal_counts: tuple[int, ...] =
                      AR_PROPOSAL_COUNTS,
                      ar_iou_grid: tuple[float, ...] = AR_IOU_GRID,
                      clip_frame_rate: Mapping[str, float] | None = None,
                      ) -> dict[str, float]:
    """Official AP/AR through the compiled evaluator, same metric keys
    as cast.eval.localization.point_estimates."""
    unknown = set(proposals.video_id) - set(labels.video_id)
    if unknown:
        raise ValueError(
            f"{len(unknown)} proposal video_ids are not in the label "
            f"table (e.g. {sorted(unknown)[:3]})")
    ev = load_evaluator()
    df = proposals.to_dataframe()
    start_s, end_s = proposal_seconds(df.video_id.to_numpy(),
                                      df.start_frame.to_numpy(),
                                      df.end_frame.to_numpy(),
                                      clip_frame_rate)
    df = df.assign(start_s=start_s.astype(np.float64),
                   end_s=end_s.astype(np.float64))
    pred: dict[str, list[list[float]]] = {str(v): [] for v in labels.video_id}
    for vid, g in df.groupby("video_id"):
        g = g.sort_values("score", ascending=False, kind="stable")
        pred[str(vid)] = [[float(r.score), float(r.start_s), float(r.end_s)]
                          for r in g.itertuples()]
    # the Rust deserializer cannot allocate an empty (n, 3) prediction
    # array; a clip with no proposals gets one zero-width placeholder at
    # the global score floor, leaving AP and AR unchanged
    for lst in pred.values():
        if not lst:
            lst.append([-1e30, 0.0, 0.0])
    meta = [{"file": str(v), "fake_segments": s.tolist()}
            for v, s in zip(labels.video_id, labels.segments)]
    with tempfile.TemporaryDirectory() as td:
        pj, mj = f"{td}/pred.json", f"{td}/meta.json"
        with open(pj, "w") as handle:
            json.dump(pred, handle)
        with open(mj, "w") as handle:
            json.dump(meta, handle)
        res = ev.ap_ar_1d(pj, mj, "file", "fake_segments", 1.0,
                          list(ap_iou_thresholds),
                          list(ar_proposal_counts),
                          list(ar_iou_grid))
    return {**{f"ap@{float(k):g}": float(v)
               for k, v in res["ap"].items()},
            **{f"ar@{int(k)}": float(v) for k, v in res["ar"].items()}}


def identity_check(python_result: dict[str, float],
                   wheel_result: dict[str, float],
                   tol: float = 1e-6) -> float:
    """Check agreement within ``tol`` (default 1e-6), metric by metric.

    Returns the maximum absolute difference; this is a tolerance check.
    """
    if not python_result:
        raise ValueError(
            "identity check ran over zero metrics; two empty results "
            "agree trivially and reporting 0.0 would claim an "
            "agreement nothing established")
    if set(python_result) != set(wheel_result):
        raise ValueError(
            f"metric key mismatch: python has "
            f"{sorted(set(python_result) - set(wheel_result))} extra, "
            f"wheel has "
            f"{sorted(set(wheel_result) - set(python_result))} extra")
    diffs = {k: abs(python_result[k] - wheel_result[k])
             for k in python_result}
    bad = {k: d for k, d in diffs.items() if not d <= tol}
    if bad:
        raise ValueError(f"identity check failed at tol={tol}: {bad}")
    return max(diffs.values()) if diffs else 0.0
