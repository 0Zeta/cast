"""Evaluate localization, detection, and bootstrap confidence intervals."""

from .bootstrap import (BootstrapResult, cluster_bootstrap, percentile_ci,
                        speaker_cluster_key)
from .localization import (AP_IOU_THRESHOLDS, AR_IOU_GRID, AR_PROPOSAL_COUNTS,
                           OFFICIAL_EVALUATOR, EvaluatorIdentity, MatchTable,
                           ap_from_multiplicities, ar_from_multiplicities,
                           evaluate_per_type, evaluate_with_identity,
                           match_proposals, point_estimates, subset_tables)
from .matching import greedy_assign, iou_matrix
from .official import evaluate_official, identity_check, load_evaluator
from .official_f32 import ARITHMETIC_IDENTITY, official_ap_from_tp, official_ar_mean

__all__ = [
    "AP_IOU_THRESHOLDS", "AR_IOU_GRID", "AR_PROPOSAL_COUNTS",
    "ARITHMETIC_IDENTITY", "OFFICIAL_EVALUATOR", "EvaluatorIdentity",
    "MatchTable", "match_proposals", "point_estimates", "subset_tables",
    "evaluate_per_type", "evaluate_with_identity",
    "ap_from_multiplicities", "ar_from_multiplicities",
    "official_ap_from_tp", "official_ar_mean",
    "iou_matrix", "greedy_assign",
    "BootstrapResult", "cluster_bootstrap", "percentile_ci", "speaker_cluster_key",
    "evaluate_official", "identity_check", "load_evaluator",
]
