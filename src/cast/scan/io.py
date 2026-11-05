"""Load detection labels, caches, and self-splice rows."""


from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Cache schema tokens: API name -> NPZ column.
SCALAR_TRACKS = {
    "cross_modal_agreement": "cos",
    "visual_continuity": "vcont",
    "audio_continuity": "acont",
    "video_side_avsr_agreement": "jv",
    "audio_side_avsr_agreement": "ja",
    "lag_advantage": "laggain",
    "visual_artifact_score": "vart",
    "audio_artifact_score": "aart",
}
SYNC_TRACKS = {
    "synchrony_residual": "sync",
}
# scores_main archive column order.
MAIN_CACHE_COLUMNS = ("cos", "vcont", "acont", "jv", "ja", "laggain")

# Self-splice corpus row kinds whose synthetic segments are localized;
# clip-global corruptions carry no localized label.
LOCALIZED_KINDS = ("a_replace", "v_replace", "av_replace", "a_shift")
BLEND_KINDS = ("blend", "ablend")

DEPLOYED_SCALAR_GROUPS = ("cross_modal_agreement",)
DEPLOYED_AVSR_GROUPS = ("audio_side_avsr_agreement",
                        "video_side_avsr_agreement")
DEPLOYED_ARTIFACT_GROUPS = ("visual_artifact_score", "audio_artifact_score")
# Extra bidirectional prediction columns are not part of the runtime score
# bank and are ignored at runtime.
DEPLOYED_SYNC_GROUPS = ("synchrony_residual",)

# Optional label columns recording a reference clip's paper-run sampling
# identity: its calibration-pool subdivision and the two sampling seeds that
# the original run derived from its original identifier.
SAMPLING_IDENTITY_COLUMNS = ("pool_split", "calibration_seed", "fusion_seed")
N_POOL_SPLITS = 4


@dataclass(frozen=True)
class ScanInputs:
    """Declared inputs of one detection run (paths resolved against
    base_dir)."""
    scores_main: str = ""
    labels_csv: str = ""
    out_dir: str = ""
    scores_sync: str | None = None
    scores_avsr: str | None = None
    scores_artifact: str | None = None
    self_splice_corpus: str | None = None
    ref_scores_main: str | None = None
    ref_labels_csv: str | None = None
    ref_scores_sync: str | None = None
    ref_scores_avsr: str | None = None
    ref_scores_artifact: str | None = None
    ref_exclude_csv: str | None = None
    subset_csv: str | None = None
    base_dir: str = ""
    main_groups: tuple = DEPLOYED_SCALAR_GROUPS
    avsr_groups: tuple = DEPLOYED_AVSR_GROUPS
    artifact_groups: tuple = DEPLOYED_ARTIFACT_GROUPS
    sync_groups: tuple = DEPLOYED_SYNC_GROUPS
    # "raise" refuses missing or incompatible rows; "skip" permits them with
    # counts and log messages.
    missing_rows: str = "raise"

    def resolve(self, key: str) -> str | None:
        """Resolve a declared path against ``base_dir``.

        Environment variables are expanded so callers can keep local data
        roots outside a checked-in input document.
        """
        v = getattr(self, key)
        if v is None or v == "":
            return None
        p = Path(os.path.expandvars(v))
        if not p.is_absolute():
            base = os.path.expandvars(self.base_dir)
            if base:
                p = Path(base) / p
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "ScanInputs":
        d = json.loads(Path(path).read_text())
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise KeyError(f"unknown ScanInputs keys {sorted(unknown)}")
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ScoreCache:
    """Read access to a scalar score cache (ragged npz: video_ids,
    lengths, offsets, scores (sum T, S), score_names)."""

    def __init__(self, npz_path):
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"score cache missing: {p}")
        with np.load(p, allow_pickle=False) as z:
            self.video_ids = z["video_ids"]
            self.lengths = z["lengths"]
            self.offsets = z["offsets"]
            self.scores = z["scores"]
            self.score_names = [str(s) for s in z["score_names"]]
        self.index = {v: i for i, v in enumerate(self.video_ids.tolist())}

    def __len__(self):
        return len(self.video_ids)

    def get_by_row(self, i):
        return self.scores[self.offsets[i]:self.offsets[i + 1]]


class SyncScoreCache:
    """Read access to the multivariate score cache."""

    def __init__(self, npz_path):
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"sync score cache missing: {p}")
        with np.load(p, allow_pickle=False) as z:
            self.video_ids = z["video_ids"]
            self.lengths = z["lengths"]
            self.offsets = z["offsets"]
            # Load declared runtime tracks and ignore extra cache blocks.
            want = tuple(SYNC_TRACKS.values())
            missing = [k for k in want if k not in z.files]
            if missing:
                raise KeyError(
                    f"sync score cache {p} is missing tracks {missing}")
            self.groups = {k: z[k] for k in want}
        self.index = {v: i for i, v in enumerate(self.video_ids.tolist())}

    def get_by_row(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return {k: g[a:b] for k, g in self.groups.items()}


def sampling_identities(labels: pd.DataFrame, source: str) -> list[tuple | None]:
    """Per-row (pool_split, calibration_seed, fusion_seed), or None.

    Genuine rows of one table either all carry an identity or none does."""
    present = [c for c in SAMPLING_IDENTITY_COLUMNS if c in labels.columns]
    if not present:
        return [None] * len(labels)
    if len(present) != len(SAMPLING_IDENTITY_COLUMNS):
        raise ValueError(f"{source}: sampling identity needs all of "
                         f"{SAMPLING_IDENTITY_COLUMNS}, found {present}")
    out = []
    for _, row in labels.iterrows():
        values = [row[c] for c in SAMPLING_IDENTITY_COLUMNS]
        blank = [v is None or (isinstance(v, float) and np.isnan(v))
                 or str(v).strip() == "" for v in values]
        if all(blank):
            out.append(None)
            continue
        if any(blank):
            raise ValueError(f"{source}: {row.video_id}: partial sampling identity")
        split = int(values[0])
        if not 0 <= split < N_POOL_SPLITS:
            raise ValueError(f"{source}: {row.video_id}: pool_split {split} "
                             f"outside 0..{N_POOL_SPLITS - 1}")
        seeds = []
        for value in values[1:]:
            text = str(value).strip()
            if len(text) != 16:
                raise ValueError(f"{source}: {row.video_id}: a sampling seed must "
                                 f"be 16 hexadecimal digits, got {text!r}")
            seeds.append(int(text, 16))
        out.append((split, *seeds))
    genuine = labels.target.to_numpy() == 0
    recorded = np.array([v is not None for v in out])
    if recorded[genuine].any() and not recorded[genuine].all():
        raise ValueError(f"{source}: some genuine rows carry a sampling "
                         "identity and others do not")
    return out


def load_clips(scores_cache, labels_csv, sync_cache_path, sync_groups,
               subset_csv=None, tag="", missing_rows="raise", log=print):
    """Clip dicts {video_id, y (T, S) float32, target, y_mv, sampling} in
    labels order. Rows absent from a cache raise unless missing_rows ==
    "skip" (counted and logged either way)."""
    cache = ScoreCache(scores_cache)
    log(f"{tag}cache: {len(cache)} clips, scores {cache.score_names}")
    sync_cache = None
    if sync_groups:
        sync_cache = SyncScoreCache(sync_cache_path)
        log(f"{tag}sync cache: {len(sync_cache.video_ids)} clips")
    labels = pd.read_csv(labels_csv)
    if subset_csv:
        keep = set(pd.read_csv(subset_csv).video_id)
        labels = labels[labels.video_id.isin(keep)]
        log(f"{tag}subset: {len(labels)} rows")
    identities = sampling_identities(labels, str(labels_csv))
    clips = []
    missing = 0
    dropped = 0
    for vid, target, sampling in zip(labels.video_id, labels.target, identities):
        i = cache.index.get(vid)
        j = sync_cache.index.get(vid) if sync_cache else None
        if i is None or (sync_cache and j is None):
            missing += 1
            continue
        y = cache.get_by_row(i)
        c = {"video_id": vid, "y": y, "target": int(target),
             "sampling": sampling}
        if sync_cache:
            mv = sync_cache.get_by_row(j)
            n_mv = len(mv[SYNC_TRACKS[sync_groups[0]]])
            # Cache join: a length disagreement of at most 2 frames
            # truncates both tracks, a larger one drops the clip.
            T = min(len(y), n_mv)
            if abs(len(y) - n_mv) > 2 or T < 2:
                dropped += 1
                continue
            c["y"] = y[:T]
            c["y_mv"] = {g: mv[SYNC_TRACKS[g]][:T] for g in sync_groups}
        clips.append(c)
    if missing:
        msg = (f"{tag}{missing} label rows not in cache "
               f"(extraction failures)")
        if missing_rows != "skip":
            raise ValueError(
                msg + "; declare missing_rows='skip' to accept the "
                      "recorded runs' skip policy")
        log(f"{tag}WARNING: {missing} label rows not in cache "
            f"(extraction failures)")
    if dropped:
        log(f"{tag}sync join: {dropped} clips dropped (length "
            f"disagreement with the primary cache beyond 2 frames)")
        if missing_rows != "skip":
            raise ValueError(
                f"{tag}{dropped} clips dropped joining the sync cache "
                f"(length disagreement beyond 2 frames); set "
                f"missing_rows='skip' to allow this")
    return clips, cache


def merge_secondary_cache(clips, scores_cache2, cache_groups,
                          missing_rows="raise", log=print):
    """Append aligned cache columns, truncating differences of at most two frames.

    missing_rows='skip' drops and counts missing or incompatible clips."""
    c2 = ScoreCache(scores_cache2)
    cols2 = [c2.score_names.index(SCALAR_TRACKS[g]) for g in cache_groups]
    merged, dropped = [], 0
    for c in clips:
        j = c2.index.get(c["video_id"])
        if j is None:
            dropped += 1
            continue
        y2 = c2.get_by_row(j)[:, cols2]
        T = min(len(c["y"]), len(y2))
        if abs(len(c["y"]) - len(y2)) > 2 or T < 2:
            dropped += 1
            continue
        c["y"] = np.concatenate([c["y"][:T], y2[:T]], axis=1)
        if "y_mv" in c:
            c["y_mv"] = {g: m[:T] for g, m in c["y_mv"].items()}
        merged.append(c)
    log(f"cache merge: {len(merged)} clips kept, {dropped} dropped")
    if dropped and missing_rows != "skip":
        raise ValueError(
            f"{dropped} clips dropped merging {scores_cache2}; declare "
            f"missing_rows='skip' to accept the recorded runs' policy")
    return merged


def load_self_splice_corpus(inputs: ScanInputs, sync_groups, log=print):
    """The self-splice corpus as clip dicts with synthetic
    segment bounds; candidate-level fitting uses rows whose kinds
    carry localized labels (genuine rows give negatives)."""
    path = inputs.resolve("self_splice_corpus")
    with np.load(path, allow_pickle=False) as archive:
        return _load_self_splice_archive(
            archive, path, inputs, sync_groups, log)


def _load_self_splice_archive(z, path, inputs, sync_groups, log):
    """Materialize the detection view while its npz archive is open."""
        # Report every detection-required key; builder-only keys are optional.
    need = ["offsets", "video_ids", "labels", "scalar"] + [
        SYNC_TRACKS[g] for g in sync_groups]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise ValueError(
            f"{path}: self-splice corpus is missing {missing}; present "
            f"keys are {sorted(z.files)}")
    offs = z["offsets"]
    vids_p = z["video_ids"]
    labels_p = z["labels"]
    scalar_p = z["scalar"]
    if inputs.avsr_groups:
        if "scalar_b" not in z.files:
            raise ValueError(
                "avsr groups need a merged self-splice corpus "
                "(scalar_b columns)")
        cols2 = [MAIN_CACHE_COLUMNS.index(SCALAR_TRACKS[g])
                 for g in inputs.avsr_groups]
        scalar_p = np.concatenate(
            [scalar_p, z["scalar_b"][:, cols2]], axis=1)
    if inputs.artifact_groups:
        if "scalar_c" not in z.files:
            raise ValueError(
                "artifact groups need a self-splice corpus with "
                "scalar_c columns")
        names_c = [str(x) for x in z["scalar_c_names"]]
        cols3 = [names_c.index(SCALAR_TRACKS[g])
                 for g in inputs.artifact_groups]
        scalar_p = np.concatenate(
            [scalar_p, z["scalar_c"][:, cols3]], axis=1)
    mv_p = {g: z[SYNC_TRACKS[g]] for g in sync_groups}
    if "seg_offsets" not in z.files:
        raise ValueError(
            f"{path}: self-splice corpus carries no segment "
            f"annotations; the candidate fusion cannot be fitted")
    seg_offs, seg_bounds, kinds_p = (z["seg_offsets"], z["seg_bounds"],
                                     z["kinds"])
    clips = []
    for j in range(len(vids_p)):
        a, b = offs[j], offs[j + 1]
        c = {"video_id": str(vids_p[j]), "y": scalar_p[a:b],
             "target": int(labels_p[j])}
        if sync_groups:
            c["y_mv"] = {g: mv_p[g][a:b] for g in sync_groups}
        kind = str(kinds_p[j])
        c["cand_fit"] = (kind == "genuine" or kind in LOCALIZED_KINDS
                         or kind in BLEND_KINDS)
        c["segs"] = seg_bounds[seg_offs[j]:seg_offs[j + 1]]
        clips.append(c)
    n_cand = sum(c["cand_fit"] for c in clips)
    log(f"self-splice corpus: {len(clips)} clips "
        f"({sum(c['target'] for c in clips)} synthetic anomalies, "
        f"{n_cand} usable for candidate fusion)")
    return clips
