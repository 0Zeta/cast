"""Execute per-clip scan stages in worker processes."""


from __future__ import annotations

import hashlib
from collections import namedtuple
from multiprocessing import Pool

import numpy as np

from .calibration import cell_ids, channel_values, sample_one_per_cell
from .channels import (clip_stat_names, clip_statistics, correction_groups,
                       frame_feature_matrix)
from .decoder import (decoder_gain, nms_proposals, segmental_decode,
                      temporal_iou)
from .fusion import (TIER2_SUFFIX, candidate_feature_matrix, mad_scale,
                     min_fusion_value)
from .interval_scan import (multivariate_candidate_vectors,
                            scalar_candidate_stats,
                            stack_prediction_candidates)

# State installed by the process-pool initializer.
_W: dict = {}


def init_worker(state):
    _W.clear()
    _W.update(state)


def clip_rng(video_id, seed, recorded=None):
    """A clip's generator: the recorded 64-bit seed when the label table
    carries one, else the first 64 bits of SHA-1("seed:video_id")."""
    if recorded is None:
        recorded = int(hashlib.sha1(f"{seed}:{video_id}".encode())
                       .hexdigest()[:16], 16)
    return np.random.default_rng(recorded)


def batched(items, size=64):
    return [items[i:i + size] for i in range(0, len(items), size)]


def pool_map(fn, state, batches, nproc):
    if nproc <= 1:
        init_worker(state)
        return [fn(b) for b in batches]
    with Pool(nproc, initializer=init_worker, initargs=(state,)) as pool:
        return pool.map(fn, batches)


# --- Candidate statistics and calibrated channel tails ---

def group_candidate_stats(y_mat, y_mv=None, excl=None):
    """Compute per-group statistics on shared candidate geometry."""
    geom_cfg = _W["geometry"]
    pooled = _W.get("pooled_baseline", False)
    geom = None
    per_group = {}
    for g, col in _W["group_cols"].items():
        null = _W["nulls"][g]
        y = null.whiten(y_mat[:, col])
        cands = scalar_candidate_stats(y, null, geom_cfg, _W["ups"][g],
                                       excl=excl, pooled_baseline=pooled)
        per_group[g] = cands
        if geom is None:
            geom = cands
    per_mv = {}
    for g in _W.get("mv_groups", ()):
        null = _W["mv_nulls"][g]
        cands = multivariate_candidate_vectors(
            null.whiten(y_mv[g]), null, geom_cfg, _W["mv_ups"][g],
            excl=excl, pooled_baseline=pooled)
        per_mv[g] = cands
        if geom is None:
            geom = cands
    return geom, per_group, per_mv


def candidate_channel_tails(geom, per_group, per_mv):
    """Calibrate candidate channels; non-finite values abstain."""
    fine, mid = cell_ids(geom.n, geom.m, geom.edge)
    rhos = {}
    for g, cands in per_group.items():
        z, zb, zt = _W["corrections"][g].apply(cands)
        ch = channel_values(z, zb, zt)
        for cname in _W["channels"]:
            vals = ch[cname]
            rho = np.full(len(vals), np.nan)
            finite = np.isfinite(vals)
            if finite.any():
                rho[finite] = _W["channel_cals"][g][cname].sf(
                    fine[finite], mid[finite], vals[finite])
            rhos[f"{g}:{cname}"] = rho
    for cg, members in correction_groups(_W.get("mv_groups", ())):
        cands = stack_prediction_candidates(per_mv, members)
        e, eb = _W["mv_corrections"][cg].energies(cands)
        for cname, vals in [(c, {"energy": e, "edge": eb}[c])
                            for c in _W["mv_channels"]]:
            rho = np.full(len(vals), np.nan)
            finite = np.isfinite(vals)
            if finite.any():
                rho[finite] = _W["mv_channel_cals"][cg][cname].sf(
                    fine[finite], mid[finite], vals[finite])
            rhos[f"{cg}:{cname}"] = rho
    return fine, mid, rhos


ScanResult = namedtuple(
    "ScanResult",
    "geom min_fused min_fused_max learned learned_max het het_max X")
_EMPTY_SCAN = ScanResult(None, None, float("nan"), None, float("nan"),
                         None, float("nan"), None)


def scan_clip(y_mat, y_mv=None, excl=None):
    """Run min, learned, and MAD-scaled candidate fusion for one clip."""
    geom, per_group, per_mv = group_candidate_stats(y_mat, y_mv, excl=excl)
    if len(geom) == 0:
        return _EMPTY_SCAN
    fine, mid, rhos = candidate_channel_tails(geom, per_group, per_mv)
    v = min_fusion_value(rhos)
    s = -np.log(np.maximum(_W["fuse_cal"].sf(fine, mid, v), 1e-300))
    s_lrn, s_clip_lrn = None, float("nan")
    s_het, s_clip_het = None, float("nan")
    X = None
    if _W.get("cand_w") is not None and _W.get("fuse_cal_lrn") is not None:
        X = candidate_feature_matrix(rhos)
        v_lrn = X @ _W["cand_w"]
        s_lrn = -np.log(np.maximum(
            _W["fuse_cal_lrn"].sf(fine, mid, v_lrn), 1e-300))
        s_clip_lrn = float(np.max(s_lrn))
        if (_W.get("cand_w_het") is not None
                and _W.get("fuse_cal_het") is not None):
            v_het = mad_scale(X) @ _W["cand_w_het"]
            s_het = -np.log(np.maximum(
                _W["fuse_cal_het"].sf(fine, mid, v_het), 1e-300))
            s_clip_het = float(np.max(s_het))
    return ScanResult(geom, s, float(np.max(s)), s_lrn, s_clip_lrn,
                      s_het, s_clip_het, X)


def decode_with_guarded_refit(y_mat, y_mv, geom, s, T, track="min_fused"):
    """Decode, then refit candidates with valid guarded complements."""
    geom_cfg = _W["geometry"]
    eps = _W["insertion_rate"]
    segs, q = segmental_decode(geom.start, geom.n, s, T, eps=eps)
    if not segs:
        return segs, q, s
    excl = np.zeros(T, dtype=bool)
    for i in segs:
        a = max(0, int(geom.start[i]) - geom_cfg.guard_band_frames)
        b = min(T, int(geom.start[i] + geom.n[i])
                + geom_cfg.guard_band_frames)
        excl[a:b] = True
    so2 = scan_clip(y_mat, y_mv, excl=excl)
    geom2 = so2.geom
    s2 = {"min_fused": so2.min_fused, "learned": so2.learned}[track]
    s_new = s.copy()
    if geom2 is not None and len(geom2) and s2 is not None:
        key1 = geom.n.astype(np.int64) * (T + 1) + geom.start
        key2 = geom2.n.astype(np.int64) * (T + 1) + geom2.start
        pos = np.searchsorted(key1, key2)
        ok = (pos < len(key1)) & (key1[np.minimum(pos, len(key1) - 1)]
                                  == key2)
        s_new[pos[ok]] = s2[ok]
    segs2, q2 = segmental_decode(geom.start, geom.n, s_new, T, eps=eps)
    return segs2, q2, s_new


def top5_proposal_mean_stat(so: ScanResult):
    """Mean of the top five NMS-filtered learned scores."""
    if so.geom is None or so.learned is None:
        return float("nan")
    keep = nms_proposals(so.geom.start, so.geom.n, so.learned,
                         iou_thr=_W["nms_iou"], top_k=5)
    return float(np.mean(so.learned[keep])) if len(keep) else float("nan")


def _clip_stats(y_mat, y_mv):
    return clip_statistics(y_mat, y_mv, _W["group_cols"], _W["nulls"],
                           _W.get("mv_groups", ()), _W["mv_nulls"])


def _frame_X(y_mat, y_mv):
    return frame_feature_matrix(y_mat, y_mv, _W["group_cols"],
                                _W["nulls"], _W.get("mv_groups", ()),
                                _W["mv_nulls"])


def worker_figure_tracks(batch):
    """Capture whole-clip scientific traces with this round's fitted objects.

    Scalars are standardized without reversing their direction. Synchrony is
    the mean squared whitened component, as in the C22 gallery. Fused scores
    retain the frame-feature calculation and its original precision.
    """
    out = []
    for vid, y_mat, y_mv, _sampling in batch:
        tracks = {g: _W["nulls"][g].whiten(y_mat[:, col])
                  for g, col in _W["group_cols"].items()}
        for g in _W.get("mv_groups", ()):
            whitened = _W["mv_nulls"][g].whiten(y_mv[g])
            tracks[g] = np.mean(np.square(whitened), axis=1)
        tracks["fused_frame_score"] = _frame_X(y_mat, y_mv) @ _W["frame_w"]
        out.append((vid, tracks))
    return out


# --- Clip feature vector and fusion statistics ---

def pvec_names(state):
    """Return the clip-fusion feature order."""
    names = ["scan"]
    if state.get("scan_cal_lrn") is not None:
        names.append("learned_scan")
    for cal, name in (("scan_cal_het", "heterogeneous_scan"),
                      ("cal_top5", "top5_proposal_mean")):
        if state.get(cal) is not None:
            names.append(name)
    return names + clip_stat_names(state["group_cols"],
                                   state.get("mv_groups", ()))


def clip_feature_vector(T, s_clip, cstats, s_clip_lrn=float("nan"),
                        extras=None):
    """Build the ordered -log(p) clip vector, mapping NaN to zero."""
    e = extras or {}
    scan_p = (_W["scan_cal"].sf(T, s_clip)
              if np.isfinite(s_clip) else float("nan"))
    ps = [scan_p]
    if _W.get("scan_cal_lrn") is not None:
        ps.append(_W["scan_cal_lrn"].sf(T, s_clip_lrn)
                  if np.isfinite(s_clip_lrn) else float("nan"))
    for cal, key in (("scan_cal_het", "s_clip_het"),
                     ("cal_top5", "top5_mean")):
        if _W.get(cal) is not None:
            v = e.get(key, float("nan"))
            ps.append(_W[cal].sf(T, v) if np.isfinite(v) else float("nan"))
    names = clip_stat_names(_W["group_cols"], _W.get("mv_groups", ()))
    ps = np.array(ps + [_W["clip_cals"][k].sf(T, cstats[k])
                        for k in names])
    x = -np.log(np.maximum(ps, 1e-300))
    x[~np.isfinite(x)] = 0.0
    return x


def fusion_statistics(T, s_clip, cstats):
    """Compute Tippett and Fisher clip-fusion statistics."""
    scan_p = (_W["scan_cal"].sf(T, s_clip)
              if np.isfinite(s_clip) else float("nan"))
    p_clip = {g: _W["clip_cals"][g].sf(T, cstats[g]) for g in cstats}
    fuse = {g: p for g, p in p_clip.items() if g.endswith(TIER2_SUFFIX)}
    ps = np.array([p for p in [scan_p, *fuse.values()] if np.isfinite(p)])
    if len(ps):
        nl = -np.log(np.maximum(ps, 1e-300))
        w_min, w_sum = float(nl.max()), float(nl.sum())
    else:
        w_min = w_sum = float("nan")
    one = ([f"{g}_low5" for g in _W["group_cols"]]
           + [f"{g}_top15" for g in _W.get("mv_groups", ())])
    ps1 = np.array([p for p in
                    [scan_p, *[p_clip[k] for k in one if k in p_clip]]
                    if np.isfinite(p)])
    w_sum1 = (float(-np.log(np.maximum(ps1, 1e-300)).sum())
              if len(ps1) else float("nan"))
    return scan_p, p_clip, w_min, w_sum, w_sum1


def _extras(so: ScanResult):
    """Return optional heterogeneous-scan and pooling statistics."""
    if _W.get("cal_top5") is None and _W.get("scan_cal_het") is None:
        return None
    return {"s_clip_het": so.het_max,
            "top5_mean": top5_proposal_mean_stat(so)}


# --- Frame tracks ---

def frame_max_track(start, n, s, T):
    """M[t] = max s over candidates covering frame t (-inf if none)."""
    from scipy.ndimage import maximum_filter1d
    M = np.full(T, -np.inf)
    for nv in np.unique(n):
        m = n == nv
        S = np.full(T, -np.inf)
        S[start[m]] = s[m]
        w = int(nv)
        out = S if w == 1 else maximum_filter1d(
            S, size=w, origin=(w - 1) // 2, mode="constant", cval=-np.inf)
        np.maximum(M, out, out=M)
    return M


def frame_inclusion_track(start, n, q, T):
    """fq[t] = sum of marginal inclusion q over candidates covering t."""
    d = np.zeros(T + 1)
    np.add.at(d, start, q)
    np.add.at(d, start + n, -q)
    return np.cumsum(d[:T])


# --- Reference-pass workers ---

def worker_channel_refs(batch):
    """Sample raw statistics for moment corrections and channel tables."""
    out = {g: [] for g in _W["group_cols"]}
    out_mv = {g: [] for g in _W.get("mv_groups", ())}
    cells_out = []
    for vid, y_mat, y_mv, sampling in batch:
        geom, per_group, per_mv = group_candidate_stats(y_mat, y_mv)
        if len(geom) == 0:
            continue
        fine, mid = cell_ids(geom.n, geom.m, geom.edge)
        sel = sample_one_per_cell(fine, clip_rng(
            vid, _W["seed"], sampling and sampling[1]))
        cells_out.append((fine[sel], mid[sel]))
        for g, cands in per_group.items():
            zt = (cands.zt0[sel] if cands.zt0 is not None
                  else np.full(len(sel), np.nan))
            out[g].append((cands.z0[sel], cands.zb0[sel], zt))
        for g, cands in per_mv.items():
            out_mv[g].append((cands.z0[sel], cands.zb0[sel]))
    if not cells_out:
        return None
    fine = np.concatenate([c[0] for c in cells_out])
    mid = np.concatenate([c[1] for c in cells_out])
    groups = {g: tuple(np.concatenate([t[i] for t in tups])
                       for i in range(3))
              for g, tups in out.items() if tups}
    mv = {g: (np.concatenate([a for a, _ in pairs]),
              np.concatenate([b for _, b in pairs]))
          for g, pairs in out_mv.items() if pairs}
    return fine, mid, groups, mv


def worker_fusion_refs(batch):
    """Sample fused values for F1 recalibration."""
    out = []
    for vid, y_mat, y_mv, sampling in batch:
        geom, per_group, per_mv = group_candidate_stats(y_mat, y_mv)
        if len(geom) == 0:
            continue
        fine, mid, rhos = candidate_channel_tails(geom, per_group, per_mv)
        v = min_fusion_value(rhos)
        if _W.get("cand_w") is not None:
            X = candidate_feature_matrix(rhos)
            v_lrn = X @ _W["cand_w"]
            v_het = (mad_scale(X) @ _W["cand_w_het"]
                     if _W.get("cand_w_het") is not None
                     else np.full(len(v), np.nan))
        else:
            v_lrn = v_het = np.full(len(v), np.nan)
        sel = sample_one_per_cell(fine, clip_rng(
            vid, _W["seed"] + 1, sampling and sampling[2]))
        out.append((v[sel], fine[sel], mid[sel], v_lrn[sel], v_het[sel]))
    if not out:
        return None
    return tuple(np.concatenate([o[i] for o in out]) for i in range(5))


def worker_frame_fit(batch):
    """Sample up to 200 positive and 200 negative frames per splice."""
    Xs, ys = [], []
    for vid, y_mat, y_mv, segs in batch:
        X = _frame_X(y_mat, y_mv)
        T = len(X)
        lab = np.zeros(T, dtype=bool)
        for a, b in segs:
            lab[a:b + 1] = True
        rng = clip_rng(vid, _W["seed"] + 3)
        pos = np.where(lab)[0]
        neg = np.where(~lab)[0]
        if len(pos) > 200:
            pos = rng.choice(pos, 200, replace=False)
        if len(neg) > 200:
            neg = rng.choice(neg, 200, replace=False)
        idx = np.concatenate([pos, neg])
        Xs.append(X[idx])
        ys.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
    if not Xs:
        return None
    return np.vstack(Xs), np.concatenate(ys)


def worker_candidate_fit(batch):
    """Sample positive and zero-overlap self-splice candidates for learned fusion weights."""
    Xs, Xhs, ys = [], [], []
    for vid, y_mat, y_mv, segs in batch:
        geom, per_group, per_mv = group_candidate_stats(y_mat, y_mv)
        if len(geom) == 0:
            continue
        fine, mid, rhos = candidate_channel_tails(geom, per_group, per_mv)
        X = candidate_feature_matrix(rhos)
        Xh = mad_scale(X)
        start = geom.start.astype(np.int64)
        end = start + geom.n.astype(np.int64) - 1
        iou = np.zeros(len(start))
        overlap = np.zeros(len(start), dtype=bool)
        for a, b in segs:
            iou = np.maximum(iou, temporal_iou(start, end, a, b))
            overlap |= (start <= b) & (end >= a)
        rng = clip_rng(vid, _W["seed"] + 2)
        pos = np.where(iou >= 0.5)[0]
        neg = np.where(~overlap)[0]
        if len(pos) > 60:
            pos = rng.choice(pos, 60, replace=False)
        if len(neg) > 100:
            neg = rng.choice(neg, 100, replace=False)
        idx = np.concatenate([pos, neg])
        Xs.append(X[idx])
        Xhs.append(Xh[idx])
        ys.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
    if not Xs:
        return None
    return np.vstack(Xs), np.vstack(Xhs), np.concatenate(ys)


def worker_scan_refs(batch):
    """Collect F2a scan, clip, pooling, and decoder statistics."""
    rows = []
    for vid, y_mat, y_mv, _sampling in batch:
        so = scan_clip(y_mat, y_mv)
        c5 = top5_proposal_mean_stat(so)
        a_gain = None
        if so.geom is not None and so.learned is not None:
            a_gain = decoder_gain(so.geom.start, so.geom.n, so.learned,
                                  len(y_mat), eps=_W["insertion_rate"])
        rows.append((len(y_mat), so.min_fused_max,
                     _clip_stats(y_mat, y_mv), so.learned_max,
                     so.het_max, c5, a_gain))
    return rows


def worker_pvec(batch):
    """-log p feature vectors for the clip fusion-weight fits."""
    rows = []
    for vid, y_mat, y_mv, _sampling in batch:
        so = scan_clip(y_mat, y_mv)
        rows.append(clip_feature_vector(
            len(y_mat), so.min_fused_max, _clip_stats(y_mat, y_mv),
            so.learned_max, _extras(so)))
    return rows


def worker_final_refs(batch):
    """Collect F2b calibration rows, proposals, and tracks."""
    rows, ref_lrn, ref_het, ref_fused = [], [], [], []
    for vid, y_mat, y_mv, _sampling in batch:
        so = scan_clip(y_mat, y_mv)
        s_clip, s_clip_lrn = so.min_fused_max, so.learned_max
        cstats = _clip_stats(y_mat, y_mv)
        _, _, w_min, w_sum, w_sum1 = fusion_statistics(len(y_mat), s_clip,
                                                       cstats)
        x = clip_feature_vector(len(y_mat), s_clip, cstats, s_clip_lrn,
                                _extras(so))
        w_lrn = (float(_W["fusion_w"] @ x)
                 if _W.get("fusion_w") is not None else float("nan"))
        w_nnls = (float(_W["nnls_w"] @ x[_W["nnls_idx"]])
                  if _W.get("nnls_w") is not None else float("nan"))
        w_expa = (float(_W["nnls_w_a"] @ x[_W["nnls_idx_a"]])
                  if _W.get("nnls_w_a") is not None else float("nan"))
        w_mix = (float(_W["mix_w"] @ x)
                 if _W.get("mix_w") is not None else float("nan"))
        rows.append((len(y_mat), w_min, w_sum, w_sum1, w_lrn, w_nnls,
                     w_expa, w_mix))
        if (so.geom is not None and so.learned is not None
                and _W.get("frame_w") is not None):
            start, n = so.geom.start, so.geom.n
            for lst, sc in ((ref_lrn, so.learned), (ref_het, so.het)):
                if sc is None:
                    continue
                keep = nms_proposals(start, n, sc, iou_thr=_W["nms_iou"],
                                     top_k=_W["top_k"])
                for rank, k in enumerate(keep):
                    lst.append((vid, int(start[k]),
                                int(start[k] + n[k] - 1),
                                float(sc[k]), rank))
            ref_fused.append((vid, (_frame_X(y_mat, y_mv)
                                    @ _W["frame_w"]).astype(np.float16)))
    return rows, ref_lrn, ref_het, ref_fused


# --- Evaluation pass ---

def worker_score(batch):
    """Score test clips and emit proposals, ranks, decodes, and tracks."""
    replay_col = _W.get("replay_col")
    stat_names = clip_stat_names(_W["group_cols"],
                                 _W.get("mv_groups", ()))
    rows, props, decoded, frames = [], [], [], []
    props_lrn, decoded_lrn, props_het = [], [], []
    gain_rows = []
    for vid, y_mat, y_mv, _sampling in batch:
        T = len(y_mat)
        replay_min = (float(y_mat[:, replay_col].min())
                      if replay_col is not None else float("nan"))
        so = scan_clip(y_mat, y_mv)
        geom, s, s_clip = so.geom, so.min_fused, so.min_fused_max
        s_lrn, s_clip_lrn = so.learned, so.learned_max
        cstats = _clip_stats(y_mat, y_mv)
        scan_p, p_clip, w_min, w_sum, w_sum1 = fusion_statistics(
            T, s_clip, cstats)
        if geom is None:
            # Prevent clip-only fusion when the scan has no candidates.
            w_min = w_sum = w_sum1 = float("nan")
        scan_p_lrn = (_W["scan_cal_lrn"].sf(T, s_clip_lrn)
                      if _W.get("scan_cal_lrn") is not None
                      and np.isfinite(s_clip_lrn) else float("nan"))
        exact_scan_lrn = (_W["scan_cal_lrn"].sf_exact(T, s_clip_lrn)
                       if _W.get("scan_cal_lrn") is not None
                       and np.isfinite(s_clip_lrn) else float("nan"))
        p_tippett = (_W["final_cal"].sf(T, w_min)
                   if np.isfinite(w_min) else float("nan"))
        p_fisher = (_W["final_cal_sum"].sf(T, w_sum)
                       if np.isfinite(w_sum) else float("nan"))
        p_fisher1 = (_W["final_cal_sum1"].sf(T, w_sum1)
                        if np.isfinite(w_sum1) else float("nan"))
        keep_lrn, segs_l, q_l, s_ref_l = None, [], None, None
        c5 = float("nan")
        fq = None
        if geom is not None and s_lrn is not None:
            start, n = geom.start, geom.n
            keep_lrn = nms_proposals(start, n, s_lrn,
                                     iou_thr=_W["nms_iou"],
                                     top_k=_W["top_k"])
            if len(keep_lrn):
                c5 = float(np.mean(s_lrn[keep_lrn[:5]]))
            segs_l, q_l, s_ref_l = decode_with_guarded_refit(
                y_mat, y_mv, geom, s_lrn, T, track="learned")
            fq = frame_inclusion_track(start, n, q_l, T)
        a_gain = exact_gain = float("nan")
        if geom is not None and s_lrn is not None:
            a_gain = decoder_gain(geom.start, geom.n, s_lrn, T,
                                  eps=_W["insertion_rate"])
            if _W.get("gain_cal") is not None and np.isfinite(a_gain):
                exact_gain = _W["gain_cal"].sf_exact(T, a_gain)
        gain_rows.append((vid, T, a_gain, exact_gain))
        extras = {"s_clip_het": so.het_max, "top5_mean": c5}
        scan_p_het = (_W["scan_cal_het"].sf(T, so.het_max)
                      if _W.get("scan_cal_het") is not None
                      and np.isfinite(so.het_max) else float("nan"))
        p_top5 = (_W["cal_top5"].sf(T, c5)
                  if _W.get("cal_top5") is not None
                  and np.isfinite(c5) else float("nan"))
        x = clip_feature_vector(T, s_clip, cstats, s_clip_lrn, extras)
        if geom is not None and _W.get("fusion_w") is not None:
            w_lrn = float(_W["fusion_w"] @ x)
            p_learned = _W["final_cal_lrn"].sf(T, w_lrn)
            exact_learned = _W["final_cal_lrn"].sf_exact(T, w_lrn)
        else:
            p_learned = exact_learned = float("nan")
        if geom is not None and _W.get("nnls_w") is not None:
            w_nnls = float(_W["nnls_w"] @ x[_W["nnls_idx"]])
            w_expa = float(_W["nnls_w_a"] @ x[_W["nnls_idx_a"]])
            p_nnls = _W["final_cal_nnls"].sf(T, w_nnls)
            exact_nnls = _W["final_cal_nnls"].sf_exact(T, w_nnls)
            p_expa = _W["final_cal_expa"].sf(T, w_expa)
            exact_expa = _W["final_cal_expa"].sf_exact(T, w_expa)
        else:
            p_nnls = exact_nnls = p_expa = exact_expa = float("nan")
        if geom is not None and _W.get("mix_w") is not None:
            w_mix = float(_W["mix_w"] @ x)
            p_mix = _W["final_cal_mix"].sf(T, w_mix)
            exact_mix = _W["final_cal_mix"].sf_exact(T, w_mix)
        else:
            p_mix = exact_mix = float("nan")
        exact_scan = (_W["scan_cal"].sf_exact(T, s_clip)
                   if np.isfinite(s_clip) else float("nan"))
        exact_final = (_W["final_cal"].sf_exact(T, w_min)
                    if np.isfinite(w_min) else float("nan"))
        exact_sum = (_W["final_cal_sum"].sf_exact(T, w_sum)
                  if np.isfinite(w_sum) else float("nan"))
        exact_sum1 = (_W["final_cal_sum1"].sf_exact(T, w_sum1)
                   if np.isfinite(w_sum1) else float("nan"))
        rows.append((vid, T, replay_min,
                     0 if geom is None else len(geom),
                     s_clip, scan_p, scan_p_lrn, scan_p_het, p_top5,
                     p_tippett, p_fisher, p_fisher1,
                     p_learned, p_nnls, p_expa, p_mix,
                     exact_scan, exact_scan_lrn, exact_final,
                     exact_sum, exact_sum1, exact_learned,
                     exact_nnls, exact_expa, exact_mix,
                     *[p_clip[k] for k in stat_names]))
        if geom is not None and _W.get("want_props"):
            start, n = geom.start, geom.n
            keep = nms_proposals(start, n, s, iou_thr=_W["nms_iou"],
                                 top_k=_W["top_k"])
            for rank, i in enumerate(keep):
                props.append((vid, int(start[i]),
                              int(start[i] + n[i] - 1),
                              float(s[i]), rank))
            segs, q, s_ref = decode_with_guarded_refit(y_mat, y_mv, geom,
                                                       s, T)
            for i in segs:
                decoded.append((vid, int(start[i]),
                                int(start[i] + n[i] - 1),
                                float(q[i]), float(s_ref[i])))
            fr = {"cross_modal_agreement":
                  (y_mat[:, replay_col] if replay_col is not None
                   else np.full(T, np.nan)),
                  "scan_max": frame_max_track(start, n, s, T),
                  "marginal_inclusion": frame_inclusion_track(start, n, q,
                                                              T),
                  "learned_scan_max": np.full(T, np.nan),
                  "learned_marginal_inclusion": np.full(T, np.nan),
                  "fused_frame_score":
                  (_frame_X(y_mat, y_mv) @ _W["frame_w"]
                   if _W.get("frame_w") is not None
                   else np.full(T, np.nan))}
            if s_lrn is not None:
                for rank, i in enumerate(keep_lrn):
                    props_lrn.append((vid, int(start[i]),
                                      int(start[i] + n[i] - 1),
                                      float(s_lrn[i]), rank))
                for i in segs_l:
                    decoded_lrn.append((vid, int(start[i]),
                                        int(start[i] + n[i] - 1),
                                        float(q_l[i]), float(s_ref_l[i])))
                fr["learned_scan_max"] = frame_max_track(start, n, s_lrn,
                                                         T)
                fr["learned_marginal_inclusion"] = fq
            if so.het is not None:
                keep_h = nms_proposals(start, n, so.het,
                                       iou_thr=_W["nms_iou"],
                                       top_k=_W["top_k"])
                for rank, i in enumerate(keep_h):
                    props_het.append((vid, int(start[i]),
                                      int(start[i] + n[i] - 1),
                                      float(so.het[i]), rank))
            frames.append((vid, {k: v.astype(np.float16)
                                 for k, v in fr.items()}))
    return (rows, props, decoded, props_lrn, decoded_lrn, frames,
            props_het, gain_rows)
