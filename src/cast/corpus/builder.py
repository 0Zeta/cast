"""Assemble matched clean and corrupted rows into a self-splice corpus."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .operators import (apply_audio_local,
                        audio_shift_generator,
                        draw_audio_local)
from .placement import DEPLOYED_PLACEMENT_LAW, PlacementLaw
from .schema import (N_SCALAR_COLUMNS, corpus_arrays,
                     corpus_from_arrays)

# Recorded artifact-column tokens (vart = visual_artifact_score,
# aart = audio_artifact_score; cast.scan.io.SCALAR_TRACKS).
ARTIFACT_COLUMN_TOKENS = ("vart", "aart")

# Source-selection rule: the first rows of each contiguous
# manifest third.
SOURCE_ROWS_PER_THIRD = 1200

# Data rows of the pinned source CSV: the 3 x 1200 selected clips minus
# the single clip lost to extraction.
SOURCE_CSV_ROWS = 3 * SOURCE_ROWS_PER_THIRD - 1

# An audio-local donor must be longer than this many frames.
MIN_DONOR_FRAMES = 30

FeatureTriple = tuple[np.ndarray, np.ndarray, np.ndarray]


def source_manifest_indices(n_manifest_rows: int,
                            per_third: int = SOURCE_ROWS_PER_THIRD,
                            ) -> list[int]:
    """Manifest row indices of the corpus source clips: rows
    [0, per_third) of each of the three contiguous thirds (third size
    ceil(n / 3))."""
    third = math.ceil(n_manifest_rows / 3)
    if per_third > min(third, n_manifest_rows - 2 * third):
        raise ValueError(
            f"cannot take {per_third} rows from each third of "
            f"{n_manifest_rows} manifest rows")
    return [i for t in range(3)
            for i in range(t * third, t * third + per_third)]


def source_csv_bytes(manifest_ids: list[str],
                     unavailable: Iterable[str] = (),
                     per_third: int = SOURCE_ROWS_PER_THIRD,
                     expected_rows: int | None = None) -> bytes:
    """Return the sorted CRLF source CSV; reject unknown exclusions and unexpected row counts."""

    lost = set(unavailable)
    sel = source_manifest_indices(len(manifest_ids), per_third)
    selected = [manifest_ids[i] for i in sel]
    if len(set(selected)) != len(selected):
        raise ValueError(
            f"{len(selected) - len(set(selected))} duplicate ids among "
            f"the selected source clips")
    unknown = sorted(lost - set(selected))
    if unknown:
        raise ValueError(
            f"{len(unknown)} ids declared unavailable are not among "
            f"the {len(selected)} selected source clips; attrition must be "
            f"matched exactly, e.g. {unknown[:3]}")
    rows = sorted(v for v in selected if v not in lost)
    if len(rows) != len(selected) - len(lost):
        raise ValueError(
            f"attrition removed {len(selected) - len(rows)} of the "
            f"{len(selected)} selected clips, expected {len(lost)}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"source csv has {len(rows)} rows, expected "
            f"{expected_rows}")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["video_id"])
    w.writerows([r] for r in rows)
    return buf.getvalue().encode()


@dataclass(frozen=True)
class ExtractionRecord:
    """One clip's re-encoded features in one feature space, with the
    realized synthetic segment bounds (inclusive frames) of the
    operator applied before encoding; segments is empty for clean
    extractions."""
    video_id: str
    audio: np.ndarray
    visual: np.ndarray
    joint: np.ndarray
    segments: tuple[tuple[int, int], ...]


class RowScorer(Protocol):
    """Score a corpus row into scalar, scalar_b, scalar_c, and sync frame groups.

    source_video_id selects the cross-fitted artifact heads."""

    def __call__(self, source_video_id: str, audio: np.ndarray,
                 visual: np.ndarray, joint: np.ndarray,
                 audio_avsr: np.ndarray, visual_avsr: np.ndarray,
                 joint_avsr: np.ndarray) -> dict[str, np.ndarray]: ...


_GROUP_KEYS = ("scalar", "scalar_b", "scalar_c", "sync")
_LEGACY_GROUP_KEYS = ("a2v", "v2a")


def build_self_splice_corpus(
        visual_blend: Iterable[ExtractionRecord],
        visual_blend_avsr: Mapping[str, ExtractionRecord],
        clean: Mapping[str, FeatureTriple],
        clean_avsr: Mapping[str, FeatureTriple],
        scorer: RowScorer, out_npz: str | Path,
        audio_blend: Iterable[ExtractionRecord] | None = None,
        audio_blend_avsr: Mapping[str, ExtractionRecord] | None = None,
        law: PlacementLaw = DEPLOYED_PLACEMENT_LAW,
        artifact_columns: tuple[str, ...] = ARTIFACT_COLUMN_TOKENS,
        seed: int = 0, limit: int = 0, log=print) -> dict[str, Any]:
    """Write the corpus NPZ and adjacent build record, then return the record.

    visual_blend order controls joins and audio-local RNG draws."""
    if (audio_blend is None) != (audio_blend_avsr is None):
        raise ValueError("audio_blend and audio_blend_avsr go together")
    rng = np.random.default_rng(seed)
    shift_rng = audio_shift_generator()
    video_ids: list[str] = []
    labels: list[int] = []
    kinds: list[str] = []
    groups: dict[str, list[np.ndarray]] = {
        k: [] for k in _GROUP_KEYS + _LEGACY_GROUP_KEYS}
    segments: list[list[tuple[int, int]]] = []
    row_records: list[dict[str, Any]] = []

    def add_row(name, label, kind, A, V, J, Ab, Vb, Jb, segs, source_vid,
                extra=None):
        T = min(len(A), len(Ab))
        A, V, J = (np.asarray(x[:T], np.float32) for x in (A, V, J))
        Ab, Vb, Jb = (np.asarray(x[:T], np.float32)
                      for x in (Ab, Vb, Jb))
        segs = [(a, min(b, T - 1)) for a, b in segs if a < T]
        scored = scorer(source_vid, A, V, J, Ab, Vb, Jb)
        missing = [k for k in _GROUP_KEYS if k not in scored]
        if missing:
            raise ValueError(f"scorer output is missing groups {missing}")
        legacy = [k for k in _LEGACY_GROUP_KEYS if k in scored]
        if legacy and tuple(legacy) != _LEGACY_GROUP_KEYS:
            raise ValueError(
                "scorer must provide both legacy prediction groups or neither")
        if groups["scalar"]:
            had_legacy = bool(groups["a2v"])
            if had_legacy != bool(legacy):
                raise ValueError(
                    "scorer changed legacy prediction-group schema between rows")
        for k in _GROUP_KEYS + tuple(legacy):
            g = np.asarray(scored[k], np.float16)
            if g.ndim != 2 or len(g) != T:
                raise ValueError(
                    f"scorer group {k}: expected ({T}, columns), got "
                    f"shape {g.shape}")
            groups[k].append(g)
        video_ids.append(name)
        labels.append(label)
        kinds.append(kind)
        segments.append(segs)
        rec = {"row_id": name, "kind": kind, "label": label,
               "source_video_id": source_vid, "length": T,
               "segments": [list(s) for s in segs]}
        if extra:
            rec.update(extra)
        row_records.append(rec)

    n_done = n_skip = 0
    prev_clean: tuple[np.ndarray, ...] | None = None
    prev_clean_vid: str | None = None
    done_vids: set[str] = set()
    seen_vids: set[str] = set()
    for rec in visual_blend:
        vid = rec.video_id
        if vid in seen_vids:
            raise ValueError(
                f"duplicate source video_id '{vid}' in visual_blend; "
                f"it would duplicate a matched pair and donor-chain entry")
        seen_vids.add(vid)
        segs = rec.segments
        rb = visual_blend_avsr.get(vid)
        if (rb is None or rb.segments != segs or not segs
                or vid not in clean or vid not in clean_avsr):
            n_skip += 1
            continue
        j = n_done
        done_vids.add(vid)
        add_row(f"blend{j:05d}", 1, "blend",
                rec.audio, rec.visual, rec.joint,
                rb.audio, rb.visual, rb.joint, segs, vid)
        add_row(f"clean{j:05d}", 0, "genuine",
                *clean[vid], *clean_avsr[vid], [], vid)
        ca, cb = clean[vid], clean_avsr[vid]
        Tc = min(len(ca[0]), len(cb[0]))
        cur = tuple(np.asarray(x[:Tc], np.float32) for x in (*ca, *cb))
        # both ends must fit: the target long enough to place a segment,
        # the donor long enough to supply one
        if (prev_clean is not None and Tc > MIN_DONOR_FRAMES
                and len(prev_clean[0]) > MIN_DONOR_FRAMES):
            kind, p, segs_a = draw_audio_local(rng, Tc,
                                               len(prev_clean[0]), law,
                                               shift_rng)
            A2, J2 = apply_audio_local(cur[0], cur[2], prev_clean[0],
                                       prev_clean[2], kind, p)
            A2b, J2b = apply_audio_local(cur[3], cur[5], prev_clean[3],
                                         prev_clean[5], kind, p)
            add_row(f"alocal{j:05d}", 1, kind,
                    A2, cur[1], J2, A2b, cur[4], J2b, segs_a, vid,
                    extra={"parameters": _json_params(p),
                           "donor_video_id": prev_clean_vid})
        prev_clean = cur
        prev_clean_vid = vid
        n_done += 1
        if n_done % 500 == 0:
            log(f"  [{n_done}] matched pairs ({n_skip} skipped)")
        if limit and n_done >= limit:
            break
    log(f"{n_done} matched pairs, {n_skip} skipped "
        f"(segment mismatch / missing counterpart)")

    n_ab = n_ab_skip = 0
    if audio_blend is not None:
        seen_ab: set[str] = set()
        for rec in audio_blend:
            vid = rec.video_id
            if vid in seen_ab:
                raise ValueError(
                    f"duplicate source video_id '{vid}' in "
                    f"audio_blend; one source clip contributes one "
                    f"audio-blend row")
            seen_ab.add(vid)
            rb = audio_blend_avsr.get(vid)
            if (vid not in done_vids or rb is None
                    or rb.segments != rec.segments or not rec.segments):
                n_ab_skip += 1
                continue
            add_row(f"ablend{n_ab:05d}", 1, "ablend",
                    rec.audio, rec.visual, rec.joint,
                    rb.audio, rb.visual, rb.joint, rec.segments, vid)
            n_ab += 1
            if limit and n_ab >= limit:
                break
        log(f"{n_ab} audio blends, {n_ab_skip} skipped")

    if not video_ids:
        raise ValueError("no corpus rows: every source clip failed the "
                         "five-way join")
    if len(artifact_columns) != groups["scalar_c"][0].shape[1]:
        raise ValueError(
            f"artifact_columns {artifact_columns} disagree with the "
            f"scorer's scalar_c width {groups['scalar_c'][0].shape[1]}")
    if groups["scalar"][0].shape[1] != N_SCALAR_COLUMNS:
        raise ValueError(
            f"scalar group must have {N_SCALAR_COLUMNS} columns, got "
            f"{groups['scalar'][0].shape[1]}")
    # validate before writing: a corpus that SelfSpliceCorpus.load would
    # reject must not leave an npz and build record behind

    stored_groups = {k: value for k, value in groups.items() if value}
    arrays = corpus_arrays(video_ids, labels, kinds, stored_groups, segments,
                           artifact_columns)
    corpus_from_arrays(arrays).validate(str(out_npz))
    np.savez(out_npz, **arrays)

    counts: dict[str, int] = {}
    for k in kinds:
        counts[k] = counts.get(k, 0) + 1
    record = {
        "seed": seed,
        "placement_law": {"mu": law.mu, "sigma": law.sigma,
                          "max_frames": law.max_frames},
        "artifact_columns": list(artifact_columns),
        "counts": counts,
        "n_segments": sum(len(s) for s in segments),
        "skipped_pairs": n_skip,
        "skipped_audio_blends": n_ab_skip,
        "rows": row_records,
    }
    record_path = Path(out_npz).with_suffix(".build.json")
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    log(f"wrote {out_npz}: {len(video_ids)} rows {counts} "
        f"(build record {record_path})")
    return record


def _json_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: (list(map(list, v)) if k == "replacements" else
                list(v) if isinstance(v, tuple) else v)
            for k, v in params.items()}
