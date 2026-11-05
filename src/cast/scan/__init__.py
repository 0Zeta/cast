"""Fit scan calibrations and emit interval proposals."""


from .calibration import (CHANNELS, MV_CHANNELS, CellCalibration,
                          ClipLengthCalibration, DirectionalCorrection,
                          MomentCorrection, TailCalibration, cell_ids,
                          channel_values, sample_one_per_cell,
                          zhang_stephens_gpd_fit)
from .channels import (candidate_channel_names, clip_stat_names,
                       clip_statistics, correction_groups,
                       frame_feature_matrix, frame_feature_names,
                       multivariate_clip_stats)
from .decoder import (decoder_gain, insertion_penalties, nms_proposals,
                      segmental_decode, temporal_iou)
from .driver import name_hash, run_detection, speaker_of
from .fusion import (candidate_feature_matrix, fit_nonneg_logistic,
                     fit_precision_weights, mad_scale, min_fusion_value,
                     mix_weights, nnls_index_sets)
from .interval_scan import (CandidateSet, MultivariateCandidateSet,
                            ScanGeometry, edge_variance_table,
                            multivariate_candidate_vectors,
                            scalar_candidate_stats,
                            stack_prediction_candidates)
from .io import ScanInputs, ScoreCache, SyncScoreCache
from .nulls import (MultivariateNullModel, ScalarNullModel,
                    fit_multivariate_null, fit_scalar_null)
from .rescoring import (temporal_contrast_gamma,
                        temporal_contrast_rescoring, track_contrasts,
                        unscaled_contrast_rescoring)

__all__ = [
    "run_detection", "ScanInputs", "ScanGeometry",
    "ScalarNullModel", "MultivariateNullModel",
    "fit_scalar_null", "fit_multivariate_null",
    "CandidateSet", "MultivariateCandidateSet",
    "scalar_candidate_stats", "multivariate_candidate_vectors",
    "stack_prediction_candidates", "edge_variance_table",
    "TailCalibration", "CellCalibration", "ClipLengthCalibration",
    "MomentCorrection", "DirectionalCorrection",
    "cell_ids", "channel_values", "sample_one_per_cell",
    "zhang_stephens_gpd_fit", "CHANNELS", "MV_CHANNELS",
    "candidate_channel_names", "clip_stat_names", "clip_statistics",
    "correction_groups", "frame_feature_matrix", "frame_feature_names",
    "multivariate_clip_stats",
    "fit_nonneg_logistic", "fit_precision_weights", "min_fusion_value",
    "candidate_feature_matrix", "mad_scale", "mix_weights",
    "nnls_index_sets",
    "segmental_decode", "decoder_gain", "nms_proposals",
    "insertion_penalties", "temporal_iou",
    "temporal_contrast_gamma", "temporal_contrast_rescoring",
    "track_contrasts", "unscaled_contrast_rescoring",
    "ScoreCache", "SyncScoreCache", "name_hash", "speaker_of",
]
