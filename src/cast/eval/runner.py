"""Select evaluation populations and run the published metric views."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import FPS, LabelTable, ProposalTable
from .bootstrap import cluster_bootstrap, percentile_ci, speaker_cluster_key
from .detection import (evaluate_detection, exact_rank_rates, panel_masks,
                        score_panel)
from .localization import (AP_IOU_THRESHOLDS, AR_IOU_GRID, AR_PROPOSAL_COUNTS,
                           match_proposals, point_estimates)
from .official import evaluate_official, identity_check
from .official_f32 import ARITHMETIC_IDENTITY

PRIMARY = "clip_pvalue"
EXACT_RANK = "exact_clip_rank"
SCORE_COLUMNS = (PRIMARY, EXACT_RANK, "fisher_pvalue", "exact_fisher_rank",
                 "scan_pvalue")
BENCHMARK_AP = (0.5, 0.75, 0.9, 0.95)
BENCHMARK_AR = (50, 30, 20, 10, 5)
COMPARISON_AP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
COMPARISON_AR = (20, 10, 5, 2)
COMPARISON_AR_GRID = tuple(round(0.1 * i, 1) for i in range(1, 8))


def localization_views() -> dict:
    """The reported benchmark and GEM-TFL grids and proposal budgets."""
    return {
        "benchmark": {"ap_iou_thresholds": list(BENCHMARK_AP),
                      "ar_iou_grid": list(AR_IOU_GRID),
                      "ar_proposal_counts": list(BENCHMARK_AR)},
        "comparison": {"ap_iou_thresholds": list(COMPARISON_AP),
                       "ar_iou_grid": list(COMPARISON_AR_GRID),
                       "ar_proposal_counts": list(COMPARISON_AR)},
    }


def reported_metrics(*, detection: bool, localization: bool) -> dict:
    """Names of the quantities reported by each enabled evaluator."""
    return {
        "detection": (["roc_auc", "average_precision", "eer", "tpr_at_fpr",
                       "fpr_at_tpr", "exact_rank_rates"] if detection else []),
        "localization": (["average_precision", "average_recall"]
                         if localization else []),
    }


def read_table(table) -> pd.DataFrame:
    frame = table.copy() if isinstance(table, pd.DataFrame) else pd.read_csv(table)
    if not frame.video_id.is_unique or frame.video_id.isna().any():
        raise ValueError("table needs unique, nonempty video_id rows")
    if frame.video_id.astype(str).str.strip().eq("").any():
        raise ValueError("table needs unique, nonempty video_id rows")
    return frame


def select_cohort(labels, cohort=None) -> pd.DataFrame:
    """Select registered IDs before evaluation, keeping label-table order."""
    labels = read_table(labels)
    if cohort is not None:
        ids = set(read_table(cohort).video_id)
        unknown = ids - set(labels.video_id)
        if unknown:
            raise ValueError(f"cohort has {len(unknown)} clips absent from labels")
        labels = labels[labels.video_id.isin(ids)].reset_index(drop=True)
    if labels.empty:
        raise ValueError("evaluation cohort is empty")
    return labels


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in
            frame[column].value_counts(dropna=False).sort_index().items()}


def detection(clip_scores, labels, *, cohort=None, strata=("modify_type",),
              bootstrap=0, seed=1, resampling="multinomial", batch_size=128,
              unit="speaker", score_columns=SCORE_COLUMNS) -> dict:
    """Evaluate scores only on the explicitly selected label population."""
    if unit not in ("speaker", "clip"):
        raise ValueError(f"unknown resampling unit {unit!r}")
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    labels = select_cohort(labels, cohort)
    scores = read_table(clip_scores)
    overlap = set(labels.columns) & set(score_columns)
    if overlap:
        raise ValueError(f"labels carry score columns {sorted(overlap)}")
    # A full emission can serve a smaller registered cohort. Select first;
    # the numerical evaluator still receives exactly aligned vectors.
    scores = scores.loc[scores.video_id.isin(labels.video_id),
                        ["video_id", *score_columns]]
    joined = labels.merge(scores, on="video_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("no scored clips in the evaluation cohort")
    missing = labels[~labels.video_id.isin(scores.video_id)]
    unit_label = ("speaker cluster via cast.eval.bootstrap.speaker_cluster_key"
                  if unit == "speaker" else "clip")
    if bootstrap:
        result = evaluate_detection(
            joined, joined[["video_id", *score_columns]], tuple(score_columns),
            primary=PRIMARY, exact_rank=EXACT_RANK, strata=tuple(strata),
            n_resamples=bootstrap, seed=seed, scheme=resampling,
            batch_size=batch_size,
            cluster_key=speaker_cluster_key if unit == "speaker" else str)
        result["bootstrap"]["unit"] = unit_label
    else:
        panels = {}
        y = joined.target.to_numpy(dtype=np.int8)
        for name, mask in panel_masks(joined, tuple(c for c in strata
                                                   if c in joined)).items():
            columns = {c: score_panel(y[mask], joined[c].to_numpy()[mask], c,
                                      require_complete=(c == PRIMARY))
                       for c in score_columns}
            head = columns[PRIMARY]
            if not head["defined"]:
                raise ValueError(f"{name}: the primary score is undefined")
            panels[name] = {
                "population": {"n_real": int((y[mask] == 0).sum()),
                               "n_fake": int((y[mask] == 1).sum())},
                "roc_auc": head["roc_auc"],
                "operating_points": head["operating_points"],
                "other_roc_auc": {c: p["roc_auc"] for c, p in columns.items()
                                  if c != PRIMARY and p["defined"]},
                "exact_rank_rates": exact_rank_rates(
                    y[mask], joined[EXACT_RANK].to_numpy()[mask]),
                "score_panels": columns}
        result = {"primary_score": PRIMARY, "exact_rank_score": EXACT_RANK,
                  "score_columns": list(score_columns),
                  "statistic": f"-log({PRIMARY})", "overall": panels.pop("overall"),
                  "strata": panels}
    return {"population": {
                "n_labels": len(labels), "n_scored": len(joined),
                "n_real": int((joined.target == 0).sum()),
                "n_fake": int((joined.target == 1).sum()),
                "attrition": {"n_missing": len(missing),
                              "by_target": counts(missing, "target"),
                              "by_modify_type": counts(missing, "modify_type")}},
            **result}


def population(labels: pd.DataFrame, scored: set[str] | None,
               unscored: str) -> tuple[pd.DataFrame, dict]:
    """Apply the declared policy; count missing rows before exclusion."""
    if unscored not in ("exclude", "empty"):
        raise ValueError("unscored must be 'exclude' or 'empty'")
    if unscored == "exclude" and scored is None:
        raise ValueError("unscored='exclude' needs clip scores")
    n_labels = len(labels)
    n_unscored = None if scored is None else int((~labels.video_id.isin(scored)).sum())
    if unscored == "exclude":
        labels = labels[labels.video_id.isin(scored)].reset_index(drop=True)
    return labels, {
        "n_labels": n_labels, "n_evaluated": len(labels),
        "n_unscored_rows": n_unscored, "unscored_rows": unscored,
        "n_genuine": int((labels.target == 0).sum()),
        "n_forged": int((labels.target == 1).sum())}


def reporting_views(point: dict, comparison_point: dict) -> dict:
    """Report each metric from the evaluator using that view's own grid."""
    benchmark = {**{f"ap@{t:g}": point[f"ap@{t:g}"] for t in BENCHMARK_AP},
                 **{f"ar@{k}": point[f"ar@{k}"] for k in BENCHMARK_AR}}
    mean_ap = float(np.mean([point[f"ap@{t:g}"] for t in BENCHMARK_AP]))
    mean_ar = float(np.mean([point[f"ar@{k}"] for k in BENCHMARK_AR]))
    comparison = {**{f"ap@{t:g}": comparison_point[f"ap@{t:g}"]
                     for t in COMPARISON_AP},
                  **{f"ar@{k}": comparison_point[f"ar@{k}"]
                     for k in COMPARISON_AR}}
    comparison["avg_map_0107"] = float(np.mean(
        [comparison_point[f"ap@{t:g}"] for t in COMPARISON_AP]))
    return {"benchmark": {**benchmark, "mean_ap": mean_ap, "mean_ar": mean_ar,
                           "score": 0.5 * (mean_ap + mean_ar)},
            "comparison": comparison}


def localization(proposals, labels, *, clip_scores=None, cohort=None,
                 unscored, frame_rate="constant", duration_column="duration",
                 strata=("modify_type",), bootstrap=0, seed=1,
                 unit="speaker", official_check=False) -> dict:
    """Run benchmark and GEM grids through identical matching and arithmetic."""
    labels_df = select_cohort(labels, cohort)
    scores = None if clip_scores is None else read_table(clip_scores)
    labels_df, described = population(
        labels_df, None if scores is None else set(scores.video_id), unscored)
    if labels_df.empty:
        raise ValueError("no clips in the localization population")
    labels = LabelTable.from_dataframe(labels_df)
    props = proposals if isinstance(proposals, ProposalTable) else ProposalTable.from_csv(proposals)
    keep = np.isin(props.video_id, labels.video_id)
    if scores is not None:
        keep &= np.isin(props.video_id, scores.video_id)
    props = ProposalTable(props.video_id[keep], props.start_frame[keep],
                          props.end_frame[keep], props.score[keep])
    if frame_rate not in ("constant", "per-clip"):
        raise ValueError("frame_rate must be 'constant' or 'per-clip'")
    clip_frame_rate = None
    if frame_rate == "per-clip":
        if scores is None:
            raise ValueError("per-clip frame rates need clip scores")
        joined = labels_df.merge(scores[["video_id", "n_frames"]], on="video_id",
                                 how="inner", validate="one_to_one")
        clip_frame_rate = dict(zip(joined.video_id,
                                   joined.n_frames / joined[duration_column]))

    def tables_for(frame):
        sub_labels = LabelTable.from_dataframe(frame)
        mask = np.isin(props.video_id, sub_labels.video_id)
        sub_props = ProposalTable(props.video_id[mask], props.start_frame[mask],
                                  props.end_frame[mask], props.score[mask])
        return {
            "benchmark": match_proposals(sub_props, sub_labels,
                                         clip_frame_rate=clip_frame_rate),
            "comparison": match_proposals(
                sub_props, sub_labels, ap_iou_thresholds=COMPARISON_AP,
                ar_proposal_counts=COMPARISON_AR, ar_iou_grid=COMPARISON_AR_GRID,
                clip_frame_rate=clip_frame_rate)}

    tables = tables_for(labels_df)
    points = {name: point_estimates(table) for name, table in tables.items()}
    evaluator = {"ap_iou_thresholds": list(AP_IOU_THRESHOLDS),
                 "ar_proposal_counts": list(AR_PROPOSAL_COUNTS),
                 "ar_iou_grid": list(AR_IOU_GRID),
                 "arithmetic": ARITHMETIC_IDENTITY,
                 "frame_rate_rule": ("per-clip n_frames / duration"
                                     if frame_rate == "per-clip" else f"constant {FPS:g} fps")}
    result = {"evaluator": evaluator,
              "view_evaluators": {
                  "benchmark": evaluator,
                  "comparison": {**evaluator,
                                 "ap_iou_thresholds": list(COMPARISON_AP),
                                 "ar_proposal_counts": list(COMPARISON_AR),
                                 "ar_iou_grid": list(COMPARISON_AR_GRID)}},
              "population": described, "point": points["benchmark"],
              "views": reporting_views(points["benchmark"], points["comparison"]),
              "strata": {}}
    target = labels_df.target.to_numpy()
    for column in strata:
        if column not in labels_df:
            continue
        values = labels_df[column].astype(str).to_numpy()
        for value in sorted(set(values[target == 1])):
            mask = ((target == 0) | (values == value) if column == "modify_type"
                    else values == value)
            sub = {n: point_estimates(t) for n, t in tables_for(labels_df[mask]).items()}
            result["strata"][f"{column}:{value}"] = {
                "definition": (f"all genuine clips plus forged {column}={value}"
                               if column == "modify_type" else f"{column}={value}"),
                "population": {"n_clips": int(mask.sum()),
                               "n_forged": int(((target == 1) & mask).sum())},
                "point": sub["benchmark"],
                "views": reporting_views(sub["benchmark"], sub["comparison"])}
    if official_check:
        checks = {}
        for name, table in tables.items():
            official = evaluate_official(
                props, labels, ap_iou_thresholds=table.ap_iou_thresholds,
                ar_proposal_counts=table.ar_proposal_counts,
                ar_iou_grid=table.ar_iou_grid, clip_frame_rate=clip_frame_rate)
            checks[name] = {"checked": True, "tolerance": 1e-6,
                            "max_absolute_difference": identity_check(points[name], official)}
        result["official_evaluator"] = checks
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if bootstrap:
        boot = cluster_bootstrap(tables, bootstrap, seed, unit=unit)
        uncertainty = {}
        for name, samples in boot.samples.items():
            uncertainty[name] = {
                "resampling": f"{boot.unit} clusters drawn with replacement",
                "n_clusters": boot.n_clusters, "n_resamples": boot.n_resamples,
                "n_degenerate_resamples": boot.n_degenerate_resamples,
                "seed": boot.seed, "confidence_level": 0.95,
                "interval_type": "percentile", "percentiles": [2.5, 97.5],
                "numpy_method": "linear", "sd_ddof": 0,
                "marginal": {k: {"point": points[name][k],
                                 "lo": percentile_ci(v)[0], "hi": percentile_ci(v)[1],
                                 "sd": float(np.std(v[np.isfinite(v)]))}
                             for k, v in samples.items()}}
        result["uncertainty"] = uncertainty["benchmark"]
        result["view_uncertainty"] = uncertainty
    return result


def export_calibration(clip_scores, labels, output, *, cohort=None) -> Path:
    """Figure 3 counts from the learned scan maximum's exact rank."""
    labels = select_cohort(labels, cohort)
    # The reference calibration export uses round-trip CSV parsing.
    scores = (clip_scores.copy() if isinstance(clip_scores, pd.DataFrame)
              else pd.read_csv(clip_scores, float_precision="round_trip"))
    scores = read_table(scores)
    joined = labels[["video_id", "target"]].merge(
        scores[["video_id", "exact_learned_scan_rank"]], on="video_id",
        how="inner", validate="one_to_one")
    p = joined.loc[joined.target == 0, "exact_learned_scan_rank"].to_numpy(np.float64)
    if not len(p) or not np.isfinite(p).all() or not ((p > 0) & (p <= 1)).all():
        raise ValueError("calibration needs finite genuine exact learned scan ranks in (0, 1]")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "rejections", "n_genuine", "rate"])
        writer.writeheader()
        for i in range(101):
            alpha = round(i / 1000, 3)
            k = int((p <= alpha + 1e-12).sum())
            writer.writerow({"alpha": f"{alpha:.3f}", "rejections": k,
                             "n_genuine": len(p), "rate": f"{k / len(p):.12f}"})
    return output


def write_result(result: dict, output) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Detection ROC endpoints can have an infinite threshold. Keep the
    # reference evaluator's values and its JSON encoding convention.
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output
