"""Write detection scores, proposals, tracks, and fitted summaries."""


from __future__ import annotations

import csv
import json
import os

import numpy as np

from .channels import clip_stat_names
from .rescoring import (temporal_contrast_gamma,
                        temporal_contrast_rescoring,
                        unscaled_contrast_rescoring)

PROPOSAL_HEADER = ["video_id", "start_frame", "end_frame", "score",
                   "rank"]
DECODED_HEADER = ["video_id", "start_frame", "end_frame", "marginal",
                  "score"]
FRAME_TRACK_KEYS = ("cross_modal_agreement", "scan_max",
                    "learned_scan_max", "marginal_inclusion",
                    "learned_marginal_inclusion", "fused_frame_score")


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _write_frame_npz(path, frames):
    lens = np.array([len(d[FRAME_TRACK_KEYS[0]]) for _, d in frames],
                    dtype=np.int64)
    offs = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    np.savez_compressed(
        path, video_ids=np.array([v for v, _ in frames]), offsets=offs,
        **{k: np.concatenate([d[k] for _, d in frames])
           for k in FRAME_TRACK_KEYS})


def write_scientific_tracks(path, frames):
    """Save selected float64 scientific traces and their actual fit rounds."""
    keys = list(frames[0][2])
    lengths = np.array([len(d[keys[0]]) for _, _, d in frames], np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    np.savez_compressed(
        path, video_ids=np.array([v for v, _, _ in frames]),
        offsets=offsets, rounds=np.array([r for _, r, _ in frames]),
        **{k: np.concatenate([d[k] for _, _, d in frames]) for k in keys})


def clip_score_columns(group_cols, mv_groups):
    """The clip_scores.csv column list, in emission order."""
    p_cols = [f"channel_pvalue_{k}"
              for k in clip_stat_names(group_cols, mv_groups)]
    return (["video_id", "n_frames", "cross_modal_agreement_min_replay",
             "n_candidates", "scan_statistic",
             "scan_pvalue", "learned_scan_pvalue",
             "heterogeneous_scan_pvalue", "top5_proposal_mean_pvalue",
             "tippett_pvalue", "fisher_pvalue", "fisher_one_stat_pvalue",
             "learned_fusion_pvalue", "nnls_pvalue", "nnls_audio_pvalue",
             "clip_pvalue",
             "exact_scan_rank", "exact_learned_scan_rank",
             "exact_tippett_rank", "exact_fisher_rank",
             "exact_fisher_one_stat_rank", "exact_learned_fusion_rank",
             "exact_nnls_rank", "exact_nnls_audio_rank",
             "exact_clip_rank"] + p_cols + ["round"])


def write_emissions(out_dir, config, group_cols, mv_groups, geometry,
                    round_info, all_rows, all_props, all_decoded,
                    all_props_lrn, all_decoded_lrn, all_props_het,
                    all_frames, all_gain, ref_props_lrn, ref_props_het,
                    ref_fused_tracks, pseudo_props_lrn,
                    pseudo_frames, log=print, fitted_dir=None):
    _write_csv(os.path.join(out_dir, "clip_scores.csv"),
               clip_score_columns(group_cols, mv_groups), all_rows)
    _write_csv(os.path.join(out_dir, "proposals_min_fusion.csv"),
               PROPOSAL_HEADER, all_props)
    _write_csv(os.path.join(out_dir, "decoded_min_fusion.csv"),
               DECODED_HEADER, all_decoded)
    if not all_props_lrn:
        raise ValueError("no learned proposals were emitted; the "
                         "learned track is the report set")
    _write_csv(os.path.join(out_dir, "proposals_learned.csv"),
               PROPOSAL_HEADER, all_props_lrn)
    _write_csv(os.path.join(out_dir, "decoded_learned.csv"),
               DECODED_HEADER, all_decoded_lrn)
    _write_csv(os.path.join(out_dir, "proposals_heterogeneous.csv"),
               PROPOSAL_HEADER, all_props_het)

    # temporal_contrast_rescoring twins, gamma from
    # the F2b reference-pool proposals
    fused_map = {v: d["fused_frame_score"] for v, d in all_frames}
    for name, plist, refp in (
            ("proposals_learned", all_props_lrn, ref_props_lrn),
            ("proposals_heterogeneous", all_props_het, ref_props_het)):
        gamma = temporal_contrast_gamma(refp, ref_fused_tracks)
        if gamma is None:
            log(f"temporal_contrast[{name}]: skipped (ref-pool "
                f"proposals {len(refp)} too few)")
            continue
        log(f"temporal_contrast[{name}]: gamma={gamma:.3f} from "
            f"{len(refp)} ref-pool genuine proposals (w=2)")
        rescored = temporal_contrast_rescoring(plist, fused_map, gamma,
                                               log=log)
        if rescored is not None:
            _write_csv(os.path.join(out_dir, f"{name}_contrast.csv"),
                       PROPOSAL_HEADER, rescored)

    # decoder gain, a diagnostic
    _write_csv(os.path.join(out_dir, "decoder_gain.csv"),
               ["video_id", "n_frames", "gain", "exact_gain_rank"],
               all_gain)
    log(f"wrote decoder_gain.csv ({len(all_gain)} rows)")

    # Ranking emission: raw score plus unscaled fused-track contrast.
    rows16 = unscaled_contrast_rescoring(all_props_lrn, fused_map,
                                         log=log)
    _write_csv(os.path.join(out_dir, "proposals_learned_rescored.csv"),
               PROPOSAL_HEADER, rows16)
    log(f"wrote proposals_learned_rescored.csv ({len(rows16)} rows)")

    # reference-pool dump: the exact objects the diagnostic
    # temporal-contrast gamma is fitted on (FrameScores schema for the
    # fused track)
    _write_csv(os.path.join(out_dir, "reference_proposals.csv"),
               PROPOSAL_HEADER, ref_props_lrn)
    rf_ids = list(ref_fused_tracks)
    rf_lens = np.array([len(ref_fused_tracks[v]) for v in rf_ids],
                       dtype=np.int64)
    rf_offs = np.zeros(len(rf_lens) + 1, dtype=np.int64)
    np.cumsum(rf_lens, out=rf_offs[1:])
    np.savez_compressed(
        os.path.join(out_dir, "reference_frame_tracks.npz"),
        video_ids=np.array(rf_ids), offsets=rf_offs,
        frame_scores=np.concatenate(
            [np.asarray(ref_fused_tracks[v], np.float32)
             for v in rf_ids]))
    log(f"wrote reference_proposals.csv ({len(ref_props_lrn)} rows) + "
        f"reference_frame_tracks.npz ({len(rf_ids)} ref clips)")

    # Emit self-splice proposals through the same output schema.
    if pseudo_props_lrn:
        _write_csv(os.path.join(out_dir, "self_splice_proposals.csv"),
                   PROPOSAL_HEADER, pseudo_props_lrn)
        pfused = {v: d["fused_frame_score"] for v, d in pseudo_frames}
        rows16p = unscaled_contrast_rescoring(pseudo_props_lrn, pfused,
                                              log=log)
        _write_csv(os.path.join(out_dir,
                                "self_splice_proposals_rescored.csv"),
                   PROPOSAL_HEADER, rows16p)
        _write_frame_npz(os.path.join(out_dir,
                                      "self_splice_frame_tracks.npz"),
                         pseudo_frames)
        log(f"wrote self_splice_proposals.csv "
            f"({len(pseudo_props_lrn)} rows) + rescored twin "
            f"({len(rows16p)} rows) + self_splice_frame_tracks.npz")

    if all_frames:
        _write_frame_npz(os.path.join(out_dir, "frame_tracks.npz"),
                         all_frames)

    meta = {"geometry": {
                "guard_band_frames": geometry.guard_band_frames,
                "min_interval_frames": geometry.min_interval_frames,
                "max_duration_frames": geometry.max_duration_frames,
                "edge_margin_frames": geometry.edge_margin_frames},
            "groups": list(group_cols), "mv_groups": list(mv_groups),
            "config": config.to_dict(),
            "fitted_dir": (os.path.relpath(fitted_dir, out_dir)
                           if fitted_dir is not None else "fitted"),
            "rounds": round_info}
    with open(os.path.join(out_dir, "fitted_models.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"wrote {len(all_rows)} clip rows, {len(all_props)} min-fusion "
        f"proposals -> {out_dir}")
