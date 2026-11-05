"""Rescore proposals by their inside-versus-flank track contrast."""


from __future__ import annotations

import numpy as np

# Flank width of the completeness contrast.
CONTRAST_FLANK = 2
# Minimum finite reference-pool rows for any rescoring fit.
MIN_FIT_ROWS = 30


def track_contrasts(props, track_map, w=CONTRAST_FLANK):
    """Return proposal scores and inside-minus-adjacent-flank track contrasts."""
    cum = {}

    def _cum(vid):
        if vid not in cum:
            x = np.asarray(track_map[vid], np.float64)
            c = np.zeros(len(x) + 1)
            np.cumsum(x, out=c[1:])
            cum[vid] = (c, len(x))
        return cum[vid]

    scores = np.array([p[3] for p in props])
    con = np.empty(len(props))
    for k, (vid, s0, s1, _sc, _rank) in enumerate(props):
        c, T = _cum(vid)
        s0, s1 = max(0, s0), min(T - 1, s1)
        inside = (c[s1 + 1] - c[s0]) / max(s1 - s0 + 1, 1)
        lo, hi = max(0, s0 - w), min(T, s1 + 1 + w)
        n_fl = (s0 - lo) + (hi - s1 - 1)
        flank = ((c[s0] - c[lo]) + (c[hi] - c[s1 + 1])) / max(n_fl, 1)
        con[k] = inside - flank
    return scores, con


def temporal_contrast_gamma(props, track_map, w=CONTRAST_FLANK):
    """Equal-variance gamma = sd(s) / sd(contrast) from genuine
    reference-pool proposals; None when the pool is
    too small."""
    if not props:
        return None
    scores, con = track_contrasts(props, track_map, w)
    m = np.isfinite(con)
    if m.sum() < MIN_FIT_ROWS:
        return None
    return float(np.std(scores[m]) / max(np.std(con[m]), 1e-9))


def rerank_rows(props, new_scores):
    """Per-clip re-rank by new score descending; rows ordered by
    ascending video_id, stable in original order within ties."""
    order = sorted(range(len(props)),
                   key=lambda k: (props[k][0], -new_scores[k]))
    out, rank, prev = [], 0, None
    for k in order:
        vid = props[k][0]
        rank = 0 if vid != prev else rank + 1
        prev = vid
        out.append((vid, props[k][1], props[k][2],
                    float(new_scores[k]), rank))
    return out


def temporal_contrast_rescoring(props, track_map, gamma,
                                w=CONTRAST_FLANK, log=print):
    """Diagnostic twin: s' = s + gamma * contrast on the fused frame
    track; returns re-ranked rows, or None when contrasts are
    unavailable."""
    if not props or gamma is None:
        return None
    scores, con = track_contrasts(props, track_map, w)
    if not np.isfinite(con).all():
        log(f"temporal_contrast: skipped (finite contrasts "
            f"{int(np.isfinite(con).sum())}/{len(con)})")
        return None
    new = scores + gamma * con
    return rerank_rows(props, new)


def unscaled_contrast_rescoring(props, fused_track_map, log=print):
    """Add finite flank contrasts and rerank within clips."""
    s, u1 = track_contrasts(props, fused_track_map)
    new = s + u1
    ok = np.isfinite(new)
    if not ok.all():
        log(f"unscaled_contrast: {int((~ok).sum())}/{len(ok)} rows "
            f"kept their raw score (missing contrast/track)")
        new = np.where(ok, new, s)
    return rerank_rows(props, new)
