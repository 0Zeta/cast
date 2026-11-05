"""Fit a detection run and score its clips."""


from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd

from ..config import CastConfig
from .calibration import (CellCalibration, ClipLengthCalibration,
                          DirectionalCorrection, MomentCorrection,
                          channel_values)
from .channels import (candidate_channel_names, clip_stat_names,
                       correction_groups, frame_feature_names)
from .emissions import write_emissions
from .fusion import (fit_nonneg_logistic, fit_precision_weights,
                     mix_weights, nnls_index_sets)
from .interval_scan import (MultivariateCandidateSet, ScanGeometry,
                            edge_variance_table)
from .io import (MAIN_CACHE_COLUMNS, N_POOL_SPLITS, SCALAR_TRACKS,
                 ScanInputs, load_clips, load_self_splice_corpus,
                 merge_secondary_cache)
from .nulls import fit_multivariate_null, fit_scalar_null
from .worker import (batched, pool_map, pvec_names, worker_candidate_fit,
                     worker_channel_refs, worker_final_refs,
                     worker_frame_fit, worker_fusion_refs, worker_pvec,
                     worker_scan_refs, worker_score, worker_figure_tracks)


def name_hash(name: str, n: int) -> int:
    """Return the full SHA-1 digest interpreted as an integer modulo n."""
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % n


def speaker_of(video_id: str) -> str:
    """Return the leftmost VoxCeleb or LRS3 speaker component, else the full ID.

    Paper-run folds use the leftmost match."""
    m = re.search(r"/(id\d+)/", video_id)
    if m:
        return m.group(1)
    m = re.search(r"/lrs3/([^/]+)/", video_id)
    if m:
        return m.group(1)
    return video_id


def _require_clustering(clips, where: str) -> None:
    """Reject speaker assignments containing only singletons or one cluster."""
    if len(clips) <= 1:
        return
    speakers = {speaker_of(c["video_id"]) for c in clips}
    if len(speakers) == len(clips):
        raise ValueError(
            f"speaker clustering over {where} produced only singleton "
            f"clusters ({len(clips)} clips); the speaker derivation "
            f"does not match these video ids")
    if len(speakers) == 1:
        raise ValueError(
            f"speaker clustering over {where}: all {len(clips)} clips use "
            f"one key {next(iter(speakers))!r}; check directory-prefix parsing")


def _items(cs):
    return [(c["video_id"], c["y"], c.get("y_mv"), c.get("sampling"))
            for c in cs]


def pool_split(clip) -> int:
    """A genuine reference clip's calibration-pool subdivision: the recorded
    paper-run value, else the full SHA-1 of its identifier modulo four."""
    if clip.get("sampling") is not None:
        return clip["sampling"][0]
    return name_hash(clip["video_id"], N_POOL_SPLITS)


def _require_seed_zero(clips, seed: int, where: str) -> int:
    """Recorded sampling identities reproduce execution seed 0 only."""
    recorded = sum(1 for c in clips if c.get("sampling") is not None)
    if recorded and seed != 0:
        raise ValueError(
            f"{where}: {recorded} clips carry a recorded sampling identity, "
            f"which reproduces execution seed 0, but the seed is {seed}")
    return recorded


def _load_all(inputs: ScanInputs, log):
    """Load the test clips, reference clips and self-splice corpus, applying
    the configured missing-rows policy."""

    sync_groups = tuple(inputs.sync_groups)
    clips, cache = load_clips(inputs.resolve("scores_main"),
                              inputs.resolve("labels_csv"),
                              inputs.resolve("scores_sync"), sync_groups,
                              inputs.resolve("subset_csv"),
                              missing_rows=inputs.missing_rows, log=log)
    groups = list(inputs.main_groups)
    if inputs.self_splice_corpus \
            and tuple(cache.score_names) != MAIN_CACHE_COLUMNS:
        # The unnamed corpus block requires the cache's recorded column order;
        # matching only its width would assign learned weights to wrong tracks.
        raise ValueError(
            f"primary cache columns {tuple(cache.score_names)} do not match "
            f"the unnamed self-splice layout {MAIN_CACHE_COLUMNS}")
    group_cols = {g: cache.score_names.index(SCALAR_TRACKS[g])
                  for g in groups}
    n_cols = len(cache.score_names)
    if inputs.scores_avsr:
        clips = merge_secondary_cache(clips, inputs.resolve("scores_avsr"),
                                      inputs.avsr_groups,
                                      missing_rows=inputs.missing_rows,
                                      log=log)
        for k, g in enumerate(inputs.avsr_groups):
            group_cols[g] = n_cols + k
        groups += list(inputs.avsr_groups)
        n_cols += len(inputs.avsr_groups)
    if inputs.scores_artifact:
        clips = merge_secondary_cache(clips,
                                      inputs.resolve("scores_artifact"),
                                      inputs.artifact_groups,
                                      missing_rows=inputs.missing_rows,
                                      log=log)
        for k, g in enumerate(inputs.artifact_groups):
            group_cols[g] = n_cols + k
        groups += list(inputs.artifact_groups)
        n_cols += len(inputs.artifact_groups)

    ref_clips = None
    if inputs.ref_scores_main:
        ref_clips, _ = load_clips(inputs.resolve("ref_scores_main"),
                                  inputs.resolve("ref_labels_csv"),
                                  inputs.resolve("ref_scores_sync"),
                                  sync_groups, tag="ref ",
                                  missing_rows=inputs.missing_rows,
                                  log=log)
        if inputs.scores_avsr:
            if not inputs.ref_scores_avsr:
                raise ValueError("scores_avsr with an external reference "
                                 "needs ref_scores_avsr (same groups)")
            ref_clips = merge_secondary_cache(
                ref_clips, inputs.resolve("ref_scores_avsr"),
                inputs.avsr_groups, missing_rows=inputs.missing_rows,
                log=log)
        if inputs.scores_artifact:
            if not inputs.ref_scores_artifact:
                raise ValueError("scores_artifact with an external "
                                 "reference needs ref_scores_artifact")
            ref_clips = merge_secondary_cache(
                ref_clips, inputs.resolve("ref_scores_artifact"),
                inputs.artifact_groups,
                missing_rows=inputs.missing_rows, log=log)
        if inputs.ref_exclude_csv:
            # No pre-filter overlap means the exclusion list is stale or miskeyed.
            excl = set(pd.read_csv(inputs.resolve("ref_exclude_csv"))
                       .video_id)
            pre_ids = set(pd.read_csv(inputs.resolve("ref_labels_csv"),
                                      usecols=["video_id"]).video_id)
            pre_overlap = len(excl & pre_ids)
            if not pre_overlap:
                raise AssertionError(
                    f"exclusion list {inputs.ref_exclude_csv} names "
                    f"{len(excl)} ids, none of them in the pre-filter "
                    f"reference population ({len(pre_ids)} clips); the list "
                    f"is stale or keyed differently")
            n0 = len(ref_clips)
            pool_overlap = sum(1 for c in ref_clips
                               if c["video_id"] in excl)
            ref_clips = [c for c in ref_clips
                         if c["video_id"] not in excl]
            log(f"ref pool: excluded {n0 - len(ref_clips)} self-splice "
                f"source clips ({len(ref_clips)} remain); exclusion "
                f"list {len(excl)} ids, {pre_overlap} in the "
                f"pre-filter reference population, "
                f"{len(excl) - pool_overlap} absent from the loaded "
                f"pool (no-ops)")
            leftover = sum(1 for c in ref_clips if c["video_id"] in excl)
            if leftover:
                raise AssertionError(
                    f"{leftover} excluded ids still in the reference "
                    f"pool after filtering")
        elif inputs.self_splice_corpus:
            raise ValueError(
                "external reference with a self-splice corpus needs "
                "ref_exclude_csv: without it the corpus source clips "
                "can serve as calibration references")

    if not inputs.self_splice_corpus:
        raise ValueError(
            "self_splice_corpus is required: the learned candidate, "
            "frame and clip fusion weights are part of the method "
            "")
    pseudo_clips = load_self_splice_corpus(inputs, sync_groups, log=log)
    return clips, groups, group_cols, ref_clips, pseudo_clips


def run_detection(inputs: ScanInputs, config: CastConfig,
                  log=print, fitted_dir=None, figure_video_ids=()):
    """Write a detection run, exact fitted state and selected scientific tracks.

    ``fitted_dir`` defaults to ``out_dir/fitted``. Figure traces use the same
    nulls and frame weights as the test clip's actual scoring round.
    """
    scan = config.scan
    n_folds = scan.n_folds
    geometry = ScanGeometry(
        guard_band_frames=scan.guard_band_frames,
        min_interval_frames=scan.min_interval_frames,
        max_duration_frames=scan.max_duration_frames,
        edge_margin_frames=scan.edge_margin_frames)
    nproc = config.execution.nproc
    seed = config.execution.seed
    channels, mv_channels = scan.scalar_channels, scan.multivariate_channels
    out_dir = inputs.resolve("out_dir")
    os.makedirs(out_dir, exist_ok=True)
    if fitted_dir is None:
        fitted_dir = os.path.join(out_dir, "fitted")
    figure_video_ids = set(figure_video_ids)

    clips, groups, group_cols, ref_clips, pseudo_clips = _load_all(inputs,
                                                                   log)
    missing = figure_video_ids - {c["video_id"] for c in clips}
    if missing:
        raise ValueError(f"figure clips absent from scored inputs: {sorted(missing)}")
    mv_groups = tuple(inputs.sync_groups)

    if ref_clips is not None:
        # External-reference protocol: every input clip is
        # pure test; fit/calib/fuse pools are speaker-hash thirds of
        # the reference pool. A single round replaces the rotation.
        _require_clustering(ref_clips, "the external reference pool")
        recorded = _require_seed_zero(ref_clips, seed, "reference pool")
        for c in clips:
            c["fold"] = -1
        pool = {0: [], 1: [], 2: []}
        for c in ref_clips:
            if c["target"] == 0:
                pool[name_hash(speaker_of(c["video_id"]), 3)].append(c)
        rounds = [(-1, pool[0], pool[1], pool[2], clips)]
        log(f"clips={len(clips)} (all test)  groups={groups}  external "
            f"ref pools fit/cal/fuse="
            + str({k: len(v) for k, v in pool.items()})
            + f"  recorded sampling identities={recorded}")
    else:
        # 4-fold validation-internal protocol.
        _require_clustering(clips, "the evaluation corpus")
        recorded = _require_seed_zero(clips, seed, "evaluation corpus")
        for c in clips:
            c["fold"] = name_hash(speaker_of(c["video_id"]), n_folds)
        reals = {f: [c for c in clips
                     if c["fold"] == f and c["target"] == 0]
                 for f in range(n_folds)}
        log(f"clips={len(clips)}  groups={groups}  reals per fold="
            + str({f: len(v) for f, v in reals.items()})
            + f"  recorded sampling identities={recorded}")
        rounds = []
        for i in range(n_folds):
            f_fit, f_cal, f_fuse = ((i + 1) % n_folds, (i + 2) % n_folds,
                                    (i + 3) % n_folds)
            rounds.append((i, reals[f_fit], reals[f_cal], reals[f_fuse],
                           [c for c in clips if c["fold"] == i]))

    all_rows, all_props, all_decoded = [], [], []
    all_props_lrn, all_decoded_lrn, all_frames = [], [], []
    all_props_het, all_gain = [], []
    ref_props_lrn, ref_props_het, ref_fused_tracks = [], [], {}
    pseudo_props_lrn, pseudo_frames = [], []
    pseudo_done = False
    round_info = {}
    figure_tracks = []
    max_T = max(len(c["y"]) for c in clips)
    if ref_clips:
        max_T = max(max_T, max(len(c["y"]) for c in ref_clips))
    kappa_n = max(750, max_T)

    for i, r_fit, r_cal, r_fuse, test in rounds:
        log(f"[round {i}] fit={len(r_fit)} calib={len(r_cal)} "
            f"fuse={len(r_fuse)} test={len(test)}")

        nulls, ups = {}, {}
        for g, col in group_cols.items():
            nulls[g] = fit_scalar_null([c["y"][:, col] for c in r_fit],
                                       max_n=kappa_n, seed=seed)
            ups[g] = edge_variance_table(nulls[g].rho,
                                         geometry.max_duration_frames,
                                         geometry.edge_margin_frames)
            log(f"[round {i}] null[{g}]: mu={nulls[g].mu:.4f} "
                f"sigma_w={nulls[g].sigma_w:.4f} psi={nulls[g].psi:.4f} "
                f"q={nulls[g].q} kappa[8]={nulls[g].kappa[8]:.2f}")
        mv_nulls, mv_ups = {}, {}
        for g in mv_groups:
            mv_nulls[g] = fit_multivariate_null(
                [c["y_mv"][g] for c in r_fit], max_n=kappa_n, seed=seed)
            mv_ups[g] = edge_variance_table(mv_nulls[g].rho,
                                            geometry.max_duration_frames,
                                            geometry.edge_margin_frames)
            log(f"[round {i}] null_mv[{g}]: p={mv_nulls[g].p} "
                f"q={mv_nulls[g].q} psi={np.round(mv_nulls[g].psi, 3)} "
                f"kappa[8]={mv_nulls[g].kappa[8]:.2f}")
        base = {"geometry": geometry, "group_cols": group_cols,
                "nulls": nulls, "ups": ups, "mv_groups": mv_groups,
                "mv_nulls": mv_nulls, "mv_ups": mv_ups, "seed": seed,
                "nms_iou": scan.nms_iou, "top_k": scan.top_k,
                "insertion_rate": scan.insertion_rate,
                "channels": channels, "mv_channels": mv_channels,
                "pooled_baseline": scan.clip_baseline == "pooled"}

        # CALIB pool: moment corrections + per-cell channel tables
        cal_batches = batched(_items(r_cal))
        refs = [r for r in pool_map(worker_channel_refs, base,
                                    cal_batches, nproc) if r]
        fine = np.concatenate([r[0] for r in refs])
        mid = np.concatenate([r[1] for r in refs])
        corrections, channel_cals = {}, {}
        for g in group_cols:
            z0 = np.concatenate([r[2][g][0] for r in refs if g in r[2]])
            zb0 = np.concatenate([r[2][g][1] for r in refs if g in r[2]])
            zt0 = np.concatenate([r[2][g][2] for r in refs if g in r[2]])
            corrections[g] = MomentCorrection.fit(z0, zb0, zt0)
            zc = (z0 - corrections[g].mu_z) / corrections[g].sd_z
            zbc = (zb0 - corrections[g].mu_zb) / corrections[g].sd_zb
            ztc = (zt0 - corrections[g].mu_zt) / corrections[g].sd_zt
            ch = channel_values(zc, zbc, ztc)
            channel_cals[g] = {}
            for cname in channels:
                finite = np.isfinite(ch[cname])
                channel_cals[g][cname] = CellCalibration(
                    fine[finite], mid[finite], ch[cname][finite],
                    min_ref=scan.min_cell_references,
                    min_exc=scan.min_tail_exceedances)
        # one pooled correction per correction group; the prediction
        # pair is fitted on the row-aligned stacked statistics
        #
        mv_corrections, mv_channel_cals = {}, {}
        for cg, members in correction_groups(mv_groups):
            def _stack(idx, refs=refs, members=members):
                cols = [np.concatenate([r[3][m][idx] for r in refs
                                        if m in r[3]]) for m in members]
                return (cols[0] if len(cols) == 1
                        else np.concatenate(cols, axis=1))
            Z0, ZB0 = _stack(0), _stack(1)
            mv_corrections[cg] = DirectionalCorrection.fit(Z0, ZB0)
            dummy = MultivariateCandidateSet(None, None, None, Z0, ZB0,
                                             None)
            e, eb = mv_corrections[cg].energies(dummy)
            mv_channel_cals[cg] = {}
            for cname, vals in [(c, {"energy": e, "edge": eb}[c])
                                for c in mv_channels]:
                finite = np.isfinite(vals)
                mv_channel_cals[cg][cname] = CellCalibration(
                    fine[finite], mid[finite], vals[finite],
                    min_ref=scan.min_cell_references,
                    min_exc=scan.min_tail_exceedances)
        state = dict(base, corrections=corrections,
                     channel_cals=channel_cals,
                     mv_corrections=mv_corrections,
                     mv_channel_cals=mv_channel_cals)
        log(f"[round {i}] channel refs per group: {len(fine)}")

        # Learned candidate weights + frame fusion weights on the
        # self-splice corpus, scored through this round's tables;
        # the F1 recalibration below restores
        # genuine-candidate validity.
        cand_clips = [c for c in pseudo_clips
                      if c.get("segs") is not None
                      and c.get("cand_fit", True)]
        if not cand_clips:
            raise ValueError("self-splice corpus holds no clips usable "
                             "for candidate fitting")
        cb = batched([(c["video_id"], c["y"], c.get("y_mv"), c["segs"])
                      for c in cand_clips])
        cf = [r for r in pool_map(worker_candidate_fit, state, cb, nproc)
              if r]
        Xc = np.vstack([r[0] for r in cf])
        yc = np.concatenate([r[2] for r in cf])
        cand_w = (np.ones(Xc.shape[1]) if scan.candidate_fusion == "equal"
                  else fit_nonneg_logistic(Xc, yc))
        cnames = candidate_channel_names(group_cols, mv_groups, channels,
                                         mv_channels)
        top = sorted(zip(cnames, cand_w), key=lambda t: -t[1])[:6]
        log(f"[round {i}] cand_w ({int(yc.sum())} pos / "
            f"{int((1 - yc).sum())} neg): "
            + ", ".join(f"{n}={w:.2f}" for n, w in top))
        cand_w_het = fit_nonneg_logistic(np.vstack([r[1] for r in cf]),
                                         yc)
        top = sorted(zip(cnames, cand_w_het), key=lambda t: -t[1])[:6]
        log(f"[round {i}] cand_w_het: "
            + ", ".join(f"{n}={w:.2f}" for n, w in top))
        ff = [r for r in pool_map(worker_frame_fit, state, cb, nproc)
              if r]
        frame_w = fit_nonneg_logistic(np.vstack([r[0] for r in ff]),
                                      np.concatenate([r[1] for r in ff]))
        log(f"[round {i}] frame_w: " + ", ".join(
            f"{n}={w:.2f}" for n, w in
            zip(frame_feature_names(group_cols, mv_groups), frame_w)))
        state = dict(state, cand_w=cand_w, cand_w_het=cand_w_het,
                     frame_w=frame_w)

        # FUSE pool sub-split at 2:1:1
        f1, f2a, f2b = [], [], []
        for c in r_fuse:
            h = pool_split(c)
            (f1 if h in (0, 1) else f2a if h == 2 else f2b).append(c)
        fu = [r for r in pool_map(worker_fusion_refs, state,
                                  batched(_items(f1)), nproc) if r]
        v_ref = np.concatenate([r[0] for r in fu])
        fine_f1 = np.concatenate([r[1] for r in fu])
        mid_f1 = np.concatenate([r[2] for r in fu])
        fuse_cal = CellCalibration(fine_f1, mid_f1, v_ref,
                                   min_ref=scan.min_cell_references,
                                   min_exc=scan.min_tail_exceedances)
        fuse_cal_lrn = CellCalibration(
            fine_f1, mid_f1, np.concatenate([r[3] for r in fu]),
            min_ref=scan.min_cell_references,
            min_exc=scan.min_tail_exceedances)
        fuse_cal_het = CellCalibration(
            fine_f1, mid_f1, np.concatenate([r[4] for r in fu]),
            min_ref=scan.min_cell_references,
            min_exc=scan.min_tail_exceedances)
        state = dict(state, fuse_cal=fuse_cal, fuse_cal_lrn=fuse_cal_lrn,
                     fuse_cal_het=fuse_cal_het)

        # F2a: scan and per-group clip tables + the decoder-gain table
        sr = [r for rws in pool_map(worker_scan_refs, state,
                                    batched(_items(f2a)), nproc)
              for r in rws]
        Ts = [r[0] for r in sr]
        scan_cal = ClipLengthCalibration(
            Ts, [r[1] for r in sr], min_ref=scan.clip_min_references,
            min_exc=scan.min_tail_exceedances)
        clip_cals = {k: ClipLengthCalibration(
            Ts, [r[2][k] for r in sr],
            min_ref=scan.clip_min_references,
            min_exc=scan.min_tail_exceedances)
            for k in clip_stat_names(group_cols, mv_groups)}
        scan_cal_lrn = ClipLengthCalibration(
            Ts, [r[3] for r in sr], min_ref=scan.clip_min_references,
            min_exc=scan.min_tail_exceedances)
        scan_cal_het = ClipLengthCalibration(
            Ts, [r[4] for r in sr], min_ref=scan.clip_min_references,
            min_exc=scan.min_tail_exceedances)
        cal_top5 = ClipLengthCalibration(
            Ts, [r[5] for r in sr], min_ref=scan.clip_min_references,
            min_exc=scan.min_tail_exceedances)
        gvals = [float("nan") if r[6] is None else r[6] for r in sr]
        gain_cal = ClipLengthCalibration(
            Ts, gvals, min_ref=scan.clip_min_references,
            min_exc=scan.min_tail_exceedances)
        log(f"[round {i}] gain_cal: "
            f"{int(np.isfinite(np.asarray(gvals)).sum())} genuine "
            f"reference gains")
        state = dict(state, scan_cal=scan_cal, clip_cals=clip_cals,
                     scan_cal_lrn=scan_cal_lrn,
                     scan_cal_het=scan_cal_het, cal_top5=cal_top5,
                     gain_cal=gain_cal)

        # NNLS precision pooling on F1 genuine -log p vectors
        pnames = pvec_names(state)
        pv1 = [x for rws in pool_map(worker_pvec, state,
                                     batched(_items(f1)), nproc)
               for x in rws]
        X1 = np.vstack(pv1)
        nnls_idx, nnls_idx_a = nnls_index_sets(pnames)
        nnls_w = fit_precision_weights(X1[:, nnls_idx], log=log,
                                       tag=f"[round {i}] ")
        nnls_w_a = fit_precision_weights(X1[:, nnls_idx_a], log=log,
                                         tag=f"[round {i}] ")
        top = sorted(zip([pnames[k] for k in nnls_idx], nnls_w),
                     key=lambda t: -t[1])[:6]
        log(f"[round {i}] nnls_w top: "
            + ", ".join(f"{n}={w:.2f}" for n, w in top))
        state = dict(state, nnls_idx=nnls_idx, nnls_w=nnls_w,
                     nnls_idx_a=nnls_idx_a, nnls_w_a=nnls_w_a)

        # Clip fusion weights on the self-splice corpus (all rows,
        # matched pairs), then the balanced mix
        pv = [x for rws in pool_map(worker_pvec, state,
                                    batched(_items(pseudo_clips)), nproc)
              for x in rws]
        X = np.vstack(pv)
        yl = np.array([c["target"] for c in pseudo_clips], dtype=float)
        fusion_w = fit_nonneg_logistic(X, yl)
        top = sorted(zip(pnames, fusion_w), key=lambda t: -t[1])[:6]
        log(f"[round {i}] fusion_w top: "
            + ", ".join(f"{n}={w:.2f}" for n, w in top))
        mix_w = mix_weights(fusion_w, nnls_w, nnls_idx, len(pnames))
        state = dict(state, fusion_w=fusion_w, mix_w=mix_w)

        # F2b: final recalibrations + the genuine reference proposals
        # and fused tracks the diagnostic contrast gamma is fitted on
        fr_out = [o for o in pool_map(worker_final_refs, state,
                                      batched(_items(f2b)), nproc) if o]
        fr = [r for o in fr_out for r in o[0]]
        ref_props_lrn.extend(p for o in fr_out for p in o[1])
        ref_props_het.extend(p for o in fr_out for p in o[2])
        ref_fused_tracks.update({v: a for o in fr_out for v, a in o[3]})
        Tf = [r[0] for r in fr]
        kw = dict(min_ref=scan.clip_min_references,
                  min_exc=scan.min_tail_exceedances)
        final_cal = ClipLengthCalibration(Tf, [r[1] for r in fr], **kw)
        final_cal_sum = ClipLengthCalibration(Tf, [r[2] for r in fr],
                                              **kw)
        final_cal_sum1 = ClipLengthCalibration(Tf, [r[3] for r in fr],
                                               **kw)
        final_cal_lrn = ClipLengthCalibration(Tf, [r[4] for r in fr],
                                              **kw)
        final_cal_nnls = ClipLengthCalibration(Tf, [r[5] for r in fr],
                                               **kw)
        final_cal_expa = ClipLengthCalibration(Tf, [r[6] for r in fr],
                                               **kw)
        final_cal_mix = ClipLengthCalibration(Tf, [r[7] for r in fr],
                                              **kw)
        state = dict(state, final_cal=final_cal,
                     final_cal_sum=final_cal_sum,
                     final_cal_sum1=final_cal_sum1,
                     final_cal_lrn=final_cal_lrn,
                     final_cal_nnls=final_cal_nnls,
                     final_cal_expa=final_cal_expa,
                     final_cal_mix=final_cal_mix,
                     want_props=True,
                     replay_col=group_cols.get("cross_modal_agreement"))

        if fitted_dir is not None:
            _dump_stages(fitted_dir, i, nulls, mv_nulls, corrections,
                         mv_corrections, fine, mid, refs, v_ref, fine_f1,
                         mid_f1, fu, sr, fr, cand_w, cand_w_het, frame_w,
                         fusion_w, nnls_w, nnls_idx, nnls_w_a,
                         nnls_idx_a, mix_w, group_cols, mv_groups)

        selected = [c for c in test if c["video_id"] in figure_video_ids]
        if selected:
            for batch in pool_map(worker_figure_tracks, state,
                                  batched(_items(selected)), nproc):
                figure_tracks.extend((vid, i, tracks) for vid, tracks in batch)

        # Test round: every clip (real + fake)
        out = pool_map(worker_score, state, batched(_items(test)), nproc)
        n_rows = 0
        for (rows, props, decoded, props_lrn, decoded_lrn, frames,
             props_het, gain_rows) in out:
            for r in rows:
                all_rows.append((*r, i))
                n_rows += 1
            all_props.extend(props)
            all_decoded.extend(decoded)
            all_props_lrn.extend(props_lrn)
            all_decoded_lrn.extend(decoded_lrn)
            all_props_het.extend(props_het)
            all_frames.extend(frames)
            all_gain.extend(gain_rows)
        log(f"[round {i}] scored {n_rows} test clips (fusion "
            f"refs={len(v_ref)}, scan refs={len(sr)}, final "
            f"refs={len(fr)})")

        # Self-splice emission: the corpus is scored exactly
        # once per protocol, through this single round's fitted objects
        # (round -1 on ext_ref, test-fold 0 on the 4-fold protocols).
        if i in (0, -1) and not pseudo_done:
            pseudo_done = True
            outp = pool_map(worker_score, state,
                            batched(_items(pseudo_clips)), nproc)
            for (rows_p, props_p, dec_p, props_lrn_p, dec_lrn_p,
                 frames_p, het_p, gain_p) in outp:
                pseudo_props_lrn.extend(props_lrn_p)
                pseudo_frames.extend(frames_p)
            log(f"[round {i}] self-splice emission: scored "
                f"{len(pseudo_clips)} corpus clips -> "
                f"{len(pseudo_props_lrn)} learned proposals")

        round_info[i] = {
            g: {"mu": nulls[g].mu, "sigma_w": nulls[g].sigma_w,
                "psi": nulls[g].psi, "q": nulls[g].q,
                "kappa_8": float(nulls[g].kappa[8]),
                "kappa_125": float(nulls[g].kappa[125])}
            for g in group_cols}
        round_info[i]["n_refs"] = {"channel": int(len(fine)),
                                   "fusion": int(len(v_ref)),
                                   "scan": len(sr), "final": len(fr)}
        round_info[i]["cand_w"] = dict(zip(
            cnames, np.round(cand_w, 4).tolist()))
        round_info[i]["cand_w_het"] = dict(zip(
            cnames, np.round(cand_w_het, 4).tolist()))
        round_info[i]["frame_w"] = dict(zip(
            frame_feature_names(group_cols, mv_groups),
            np.round(frame_w, 4).tolist()))
        round_info[i]["nnls_w"] = dict(zip(
            [pnames[k] for k in nnls_idx], np.round(nnls_w, 4).tolist()))
        round_info[i]["nnls_w_audio"] = dict(zip(
            [pnames[k] for k in nnls_idx_a],
            np.round(nnls_w_a, 4).tolist()))
        round_info[i]["fusion_w"] = dict(zip(
            pnames, np.round(fusion_w, 4).tolist()))

    write_emissions(
        out_dir, config, group_cols, mv_groups, geometry, round_info,
        all_rows, all_props, all_decoded, all_props_lrn, all_decoded_lrn,
        all_props_het, all_frames, all_gain,
        ref_props_lrn, ref_props_het, ref_fused_tracks,
        pseudo_props_lrn, pseudo_frames, log=log, fitted_dir=fitted_dir)
    if figure_tracks:
        from .emissions import write_scientific_tracks
        write_scientific_tracks(os.path.join(out_dir, "scientific_tracks.npz"),
                                figure_tracks)
    return out_dir


def _dump_stages(dump_dir, round_id, nulls, mv_nulls, corrections,
                 mv_corrections, fine, mid, refs, v_ref, fine_f1, mid_f1,
                 fu, sr, fr, cand_w, cand_w_het, frame_w, fusion_w,
                 nnls_w, nnls_idx, nnls_w_a, nnls_idx_a, mix_w,
                 group_cols, mv_groups):
    """Write per-stage reproduction intermediates: fitted
    nulls, moment corrections, sampled reference statistics, fusion
    reference values, clip-table reference rows and every weight
    vector, one npz per stage."""
    os.makedirs(dump_dir, exist_ok=True)
    tag = f"round{round_id}"
    d = {}
    for g in group_cols:
        n = nulls[g]
        d.update({f"{g}_mu": n.mu, f"{g}_sigma_w": n.sigma_w,
                  f"{g}_psi": n.psi, f"{g}_q": n.q, f"{g}_rho": n.rho,
                  f"{g}_kappa": n.kappa})
        c = corrections[g]
        d.update({f"{g}_corr": np.array([c.mu_z, c.sd_z, c.mu_zb,
                                         c.sd_zb, c.mu_zt, c.sd_zt])})
    for g in mv_groups:
        n = mv_nulls[g]
        d.update({f"{g}_mu": n.mu, f"{g}_W": n.W, f"{g}_B": n.B,
                  f"{g}_psi": n.psi, f"{g}_q": n.q, f"{g}_rho": n.rho,
                  f"{g}_kappa": n.kappa})
    for cg in mv_corrections:
        c = mv_corrections[cg]
        d.update({f"{cg}_corr_mu_z": c.mu_z, f"{cg}_corr_Wc_z": c.Wc_z,
                  f"{cg}_corr_mu_zb": c.mu_zb,
                  f"{cg}_corr_Wc_zb": c.Wc_zb})
    np.savez_compressed(os.path.join(dump_dir, f"{tag}_models.npz"), **d)
    d2 = {"fine": fine, "mid": mid}
    for g in group_cols:
        for j, nm in enumerate(("z0", "zb0", "zt0")):
            d2[f"{g}_{nm}"] = np.concatenate(
                [r[2][g][j] for r in refs if g in r[2]])
    for g in mv_groups:
        d2[f"{g}_z0"] = np.concatenate(
            [r[3][g][0] for r in refs if g in r[3]])
        d2[f"{g}_zb0"] = np.concatenate(
            [r[3][g][1] for r in refs if g in r[3]])
    np.savez_compressed(os.path.join(dump_dir, f"{tag}_channel_refs.npz"),
                        **d2)
    np.savez_compressed(
        os.path.join(dump_dir, f"{tag}_fusion_refs.npz"),
        v=v_ref, fine=fine_f1, mid=mid_f1,
        v_lrn=np.concatenate([r[3] for r in fu]),
        v_het=np.concatenate([r[4] for r in fu]))
    np.savez_compressed(
        os.path.join(dump_dir, f"{tag}_clip_refs.npz"),
        scan_T=np.array([r[0] for r in sr]),
        scan_max=np.array([r[1] for r in sr]),
        learned_max=np.array([r[3] for r in sr]),
        het_max=np.array([r[4] for r in sr]),
        top5=np.array([r[5] for r in sr]),
        gain=np.array([float("nan") if r[6] is None else r[6]
                       for r in sr]),
        final=np.array([[r[j] for j in range(8)] for r in fr]))
    np.savez_compressed(
        os.path.join(dump_dir, f"{tag}_weights.npz"),
        cand_w=cand_w, cand_w_het=cand_w_het, frame_w=frame_w,
        fusion_w=fusion_w, nnls_w=nnls_w, nnls_idx=nnls_idx,
        nnls_w_audio=nnls_w_a, nnls_idx_audio=nnls_idx_a, mix_w=mix_w)
