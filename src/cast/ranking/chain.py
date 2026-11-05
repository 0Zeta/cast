"""Filter, feature, score, and rank emitted proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import SchemaError, _require
from ..config import RankingConfig
from .boundary_store import BoundaryCueStore
from .ranker import ProposalRanker

# duplicate-merge key: 17 bits per frame coordinate, enough for 750-frame
# clips, with the clip index in the remaining high bits
_COORD_BITS = 17
_CLIP_SHIFT = 2 * _COORD_BITS

#: features the chain computes itself; every other ranker feature is a
#: column of the boundary cue store
CHAIN_FEATURES = ("log1p_rank", "adjusted_score")


@dataclass(frozen=True)
class EmissionRun:
    """Emission manifest and proposals in stable clip-index order.

    Call reorder_to_labels before attaching label-ordered boundary cue rows."""

    video_ids: np.ndarray        # (C,) clip order of this run
    clip_index: np.ndarray       # (P,) into video_ids, nondecreasing
    start_frame: np.ndarray      # (P,) inclusive frames
    end_frame: np.ndarray        # (P,)
    score: np.ndarray            # (P,) float64 emitted working score


def _manifest_ids(clip_manifest) -> np.ndarray:
    if isinstance(clip_manifest, (str, Path)):
        ids = pd.read_csv(clip_manifest, usecols=["video_id"])["video_id"]
        _require(bool(ids.notna().all()) and
                 bool(ids.astype(str).str.strip().ne("").all()),
                 f"{clip_manifest}: blank or missing video_id")
        vids = ids.to_numpy(dtype=object)
    else:
        vids = np.array([str(v) for v in clip_manifest], dtype=object)
    _require(len(set(vids.tolist())) == len(vids),
             "duplicate video_id in the clip manifest")
    return vids


def load_emission_run(proposals_csv: str | Path, clip_manifest,
                      proposal_columns: Mapping[str, str] | None = None,
                      ) -> EmissionRun:
    """Load a proposal CSV against its clip manifest: a CSV with a
    ``video_id`` column (the emitted clip scores) or a sequence of ids.

    The default pandas float parser reproduces paper-run values."""
    cols = {"video_id": "video_id", "start_frame": "start_frame",
            "end_frame": "end_frame", "score": "score"}
    if proposal_columns:
        unknown = set(proposal_columns) - set(cols)
        _require(not unknown, f"unknown proposal column keys {sorted(unknown)}")
        cols.update(proposal_columns)
    vids = _manifest_ids(clip_manifest)
    props = pd.read_csv(proposals_csv)
    missing = [c for c in cols.values() if c not in props.columns]
    _require(not missing, f"{proposals_csv}: missing columns {missing}")
    if len(props) == 0:
        return EmissionRun(video_ids=vids,
                           clip_index=np.zeros(0, dtype=np.int64),
                           start_frame=np.zeros(0, dtype=np.int64),
                           end_frame=np.zeros(0, dtype=np.int64),
                           score=np.zeros(0, dtype=np.float64))
    vidx = {v: i for i, v in enumerate(vids)}
    pvids = props[cols["video_id"]]
    unknown_vids = ~pvids.isin(vidx)
    if unknown_vids.any():
        raise SchemaError(
            f"{proposals_csv}: {int(unknown_vids.sum())} proposals "
            f"reference clips absent from the manifest")
    ci = pvids.map(vidx).to_numpy().astype(np.int64)
    start = props[cols["start_frame"]].to_numpy()
    end = props[cols["end_frame"]].to_numpy()
    score = props[cols["score"]].to_numpy(dtype=np.float64)
    _require(np.issubdtype(start.dtype, np.integer)
             and np.issubdtype(end.dtype, np.integer),
             f"{proposals_csv}: start/end frames must be integers")
    _require(bool((start >= 0).all() and (end >= start).all()),
             f"{proposals_csv}: malformed proposal intervals")
    _require(bool(np.isfinite(score).all()),
             f"{proposals_csv}: non-finite proposal score")
    order = np.argsort(ci, kind="stable")
    return EmissionRun(video_ids=vids, clip_index=ci[order],
                       start_frame=start.astype(np.int64)[order],
                       end_frame=end.astype(np.int64)[order],
                       score=score[order])


def scored_clip_order(labels_csv: str | Path, emitted) -> list[str]:
    """Label-table clip order restricted to the emitted clips.

    Label rows without scores are declared attrition; an emitted clip
    missing from the labels is refused."""
    ids = pd.read_csv(labels_csv, usecols=["video_id"])["video_id"]
    _require(bool(ids.notna().all()) and
             bool(ids.astype(str).str.strip().ne("").all()),
             f"{labels_csv}: blank or missing video_id")
    _require(bool(ids.is_unique), f"{labels_csv}: duplicate video_id rows")
    emitted = {str(v) for v in emitted}
    unknown = emitted - set(ids)
    _require(not unknown, f"{labels_csv}: lacks {len(unknown)} emitted clips, "
                          f"e.g. {sorted(unknown)[:3]}")
    return ids[ids.isin(emitted)].tolist()


def reorder_to_labels(run: EmissionRun, labels_video_ids) -> EmissionRun:
    """Stably reorder clip blocks to label-table order without changing per-clip content."""
    order = [v for v in labels_video_ids]
    if sorted(order) != sorted(run.video_ids):
        raise SchemaError(
            "label-table clip order is not a permutation of the "
            "emission manifest; cannot align the boundary cue store")
    pos = {v: i for i, v in enumerate(run.video_ids)}
    old_of_new = np.array([pos[v] for v in order], dtype=np.int64)
    new_of_old = np.empty(len(old_of_new), dtype=np.int64)
    new_of_old[old_of_new] = np.arange(len(old_of_new))
    nci = new_of_old[run.clip_index]
    perm = np.argsort(nci, kind="stable")
    return EmissionRun(
        video_ids=np.array(order, dtype=object),
        clip_index=nci[perm],
        start_frame=run.start_frame[perm],
        end_frame=run.end_frame[perm],
        score=run.score[perm])


def min_proposal_width_filter(start_frame: np.ndarray,
                              end_frame: np.ndarray,
                              min_width: int) -> np.ndarray:
    """Keep mask for proposals of inclusive width >= min_width."""
    return (end_frame - start_frame + 1) >= min_width


def duplicate_merge(clip_index: np.ndarray, start_frame: np.ndarray,
                    end_frame: np.ndarray,
                    score: np.ndarray) -> np.ndarray:
    """Keep the first highest-scoring row for each interval."""
    _require(bool((start_frame >= 0).all() and (end_frame >= 0).all()),
             "negative frame index in the dedup key")
    _require(bool((start_frame < 2 ** _COORD_BITS).all()
                  and (end_frame < 2 ** _COORD_BITS).all()),
             f"frame index exceeds the {_COORD_BITS}-bit dedup-key capacity")
    _require(bool((clip_index >= 0).all())
             and (len(clip_index) == 0
                  or int(clip_index.max()) < 2 ** (63 - _CLIP_SHIFT)),
             "clip_index exceeds the dedup-key capacity")
    key = (clip_index.astype(np.int64) << _CLIP_SHIFT) \
        + (start_frame.astype(np.int64) << _COORD_BITS) \
        + end_frame.astype(np.int64)
    order = np.lexsort((-score, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    keep = np.zeros(len(order), dtype=bool)
    keep[order[first]] = True
    return keep


def within_clip_resort(clip_index: np.ndarray,
                       sort_score: np.ndarray) -> np.ndarray:
    """Permutation sorting each clip's proposals by score, descending and
    stable, with clip blocks contiguous in ascending clip_index order."""
    return np.lexsort((-sort_score, clip_index))


def within_clip_ranks(clip_index: np.ndarray) -> np.ndarray:
    """0-based within-clip ranks of a table with contiguous clip blocks."""
    n = len(clip_index)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    block_start = np.flatnonzero(
        np.r_[True, clip_index[1:] != clip_index[:-1]])
    blocks = clip_index[block_start]
    _require(len(np.unique(blocks)) == len(blocks),
             "within_clip_ranks needs contiguous clip blocks")
    sizes = np.diff(np.r_[block_start, n])
    return np.arange(n) - np.repeat(block_start, sizes)


@dataclass(frozen=True)
class PreparedProposals:
    """Proposals after the width filter, duplicate merge, and within-clip
    resort: the rows the boundary cue store and the ranker are aligned to."""

    clip_index: np.ndarray
    start_frame: np.ndarray
    end_frame: np.ndarray
    score: np.ndarray
    rank: np.ndarray             # 0-based within-clip rank by score

    def __len__(self) -> int:
        return len(self.clip_index)


def prepare_proposals(run: EmissionRun,
                      config: RankingConfig = RankingConfig(),
                      ) -> PreparedProposals:
    """Stages 1-3 of the chain: width floor, duplicate merge, resort."""
    ci, start, end, score = (run.clip_index, run.start_frame,
                             run.end_frame, run.score)
    keep = min_proposal_width_filter(start, end, config.min_proposal_width)
    ci, start, end, score = ci[keep], start[keep], end[keep], score[keep]
    keep = duplicate_merge(ci, start, end, score)
    ci, start, end, score = ci[keep], start[keep], end[keep], score[keep]
    perm = within_clip_resort(ci, score)
    ci, start, end, score = ci[perm], start[perm], end[perm], score[perm]
    return PreparedProposals(clip_index=ci, start_frame=start, end_frame=end,
                             score=score, rank=within_clip_ranks(ci))


def design_matrix(prepared: PreparedProposals,
                  feature_names: Sequence[str],
                  boundary_cues: BoundaryCueStore | None, *,
                  require_measured: bool = True) -> np.ndarray:
    """Ranker features in the model's column order: chain features from
    the prepared table, every other name from the boundary cue store."""
    chain = {"log1p_rank": np.log1p(prepared.rank),
             "adjusted_score": prepared.score.astype(np.float64)}
    names = tuple(feature_names)
    _require(len(set(names)) == len(names) and len(names) > 0,
             "ranker features must be a nonempty list of distinct names")
    cue_names = tuple(c for c in names if c not in chain)
    cue = None
    if cue_names:
        if boundary_cues is None:
            raise SchemaError(f"features {list(cue_names)} need a boundary cue store")
        unknown = [c for c in cue_names if c not in boundary_cues.column_names]
        if unknown:
            raise SchemaError(
                f"ranker features {unknown} are neither chain features "
                f"{list(CHAIN_FEATURES)} nor boundary cue columns "
                f"{list(boundary_cues.column_names)}")
        cue = boundary_cues.columns(cue_names, prepared.start_frame,
                                    prepared.end_frame,
                                    require_measured=require_measured)
    cue_at = {c: k for k, c in enumerate(cue_names)}
    return np.column_stack([chain[c] if c in chain else cue[:, cue_at[c]]
                            for c in names])


def posterior_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Per clip, sort by score descending (stable) and recompute 0-based ranks."""
    out = df.sort_values(["video_id", "score"], ascending=[True, False],
                         kind="stable")
    out["rank"] = out.groupby("video_id", sort=False).cumcount()
    return out


def run_ranking_chain(run: EmissionRun, ranker: ProposalRanker,
                      boundary_cues: BoundaryCueStore | None,
                      config: RankingConfig = RankingConfig(),
                      ) -> pd.DataFrame:
    """Score every prepared proposal with the ranker and rank it within
    its clip; no clip-level gate is applied."""
    prepared = prepare_proposals(run, config)
    X = design_matrix(prepared, ranker.feature_names, boundary_cues)
    post = ranker.score(X, ranker.feature_names)
    df = pd.DataFrame({"video_id": run.video_ids[prepared.clip_index],
                       "start_frame": prepared.start_frame,
                       "end_frame": prepared.end_frame,
                       "score": post})
    return posterior_ranking(df)
