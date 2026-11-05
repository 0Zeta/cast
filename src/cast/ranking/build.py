"""Build proposal audio boundary cues, fit the ranker, and write rankings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import RankingConfig
from ..corpus.audio import (decode_audio, donor_sources, media_paths,
                            spliced_waveform)
from ..corpus.schema import SelfSpliceCorpus
from .audio_boundary import (SAMPLE_RATE, STFT_HOP, column_names, curves,
                            extract_clip, proposal_features)
from .boundary_store import BoundaryCueStore
from .chain import (design_matrix, load_emission_run, prepare_proposals,
                    reorder_to_labels, run_ranking_chain, scored_clip_order)
from .labels import ASSIGNMENT_IOU, synthetic_segment_labels
from .ranker import FINAL_PROPOSAL_FEATURES, ITERATIONS, RIDGE, ProposalRanker


def build_boundary_cues(proposals, clip_scores, labels, manifest, output, *,
                       curves_dir=None, curve_video_ids=None) -> Path:
    """Build cues in scored label order, optionally saving selected clip curves.

    Curve files are <clip_index>.npz in that same order. Each carries scalar
    video_id and 1-D time_seconds and full_band_flux_percentile arrays. Time
    marks STFT step starts. Capture uses the curve already computed for cues.
    With curves_dir and no selection, capture every emitted clip.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    run = load_emission_run(proposals, clip_scores)
    run = reorder_to_labels(run, scored_clip_order(labels, run.video_ids))
    paths = media_paths(manifest)
    missing = [str(v) for v in run.video_ids if str(v) not in paths]
    if missing:
        raise KeyError(f"manifest lacks {len(missing)} emitted clips, e.g. {missing[:3]}")
    selected = set()
    curve_paths = {}
    if curve_video_ids is not None and curves_dir is None:
        raise ValueError("curve_video_ids requires curves_dir")
    if curves_dir is not None:
        selected = (set(run.video_ids) if curve_video_ids is None
                    else {str(v) for v in curve_video_ids})
        unknown = selected - set(run.video_ids)
        if unknown:
            raise KeyError(f"curve selection contains unemitted clips: {sorted(unknown)[:3]}")
        curve_paths = {i: Path(curves_dir) / f"{i}.npz"
                       for i, v in enumerate(run.video_ids) if v in selected}
        for path in curve_paths.values():
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")

    prepared = prepare_proposals(run, RankingConfig())
    features = []
    position = 0
    for clip, video_id in enumerate(run.video_ids):
        rows = np.flatnonzero(prepared.clip_index == clip)
        if not len(rows) and video_id not in selected:
            continue
        ranked = curves(decode_audio(paths[str(video_id)]))
        if len(rows):
            features.append(proposal_features(ranked, prepared.start_frame[rows],
                                              prepared.end_frame[rows]))
            position += 1
            if position % 100 == 0:
                print(f"boundary cues: {position} clips", flush=True)
        if clip in curve_paths:
            path = curve_paths[clip]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, video_id=np.asarray(str(video_id)),
                     time_seconds=np.arange(len(ranked)) * STFT_HOP / SAMPLE_RATE,
                     full_band_flux_percentile=ranked[:, 0])
    flat = (np.concatenate(features) if features
            else np.zeros((0, len(column_names())), np.float32))
    store = BoundaryCueStore(start_frame=prepared.start_frame,
                             end_frame=prepared.end_frame,
                             features=flat.astype(np.float64),
                             column_names=column_names())
    output.parent.mkdir(parents=True, exist_ok=True)
    store.save(output)
    print(f"wrote {output}: {len(store)} proposal rows")
    return output


def source_ids(build_record: str | Path, corpus: SelfSpliceCorpus) -> list[str]:
    """The source clip of every corpus row, from the corpus build record."""
    rows = json.loads(Path(build_record).read_text())["rows"]
    if [r["row_id"] for r in rows] != corpus.video_ids.tolist():
        raise ValueError(f"{build_record} does not describe this corpus row for row")
    return [str(r["source_video_id"]) for r in rows]


def build_self_splice_boundary_cues(proposals, corpus, manifest, output, *,
                                   build_record=None) -> Path:
    """Rebuild cues using corpus source mappings and realized segment bounds."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    record = build_record or str(Path(corpus).with_suffix(".build.json"))
    corpus = SelfSpliceCorpus.load(corpus)
    sources = source_ids(record, corpus)
    donors = donor_sources(corpus.kinds.tolist(), sources)
    paths = media_paths(manifest)
    missing = sorted({s for s in sources if s not in paths})
    if missing:
        raise KeyError(f"manifest lacks {len(missing)} source clips, e.g. {missing[:3]}")

    run = load_emission_run(proposals, corpus.video_ids.tolist())
    prepared = prepare_proposals(run, RankingConfig())
    features = np.full((len(prepared), len(column_names())), np.nan, np.float32)
    cache: dict[str, np.ndarray | None] = {}

    def waveform(video_id: str) -> np.ndarray | None:
        if video_id not in cache:
            if len(cache) > 200:
                cache.clear()
            try:
                cache[video_id] = decode_audio(paths[video_id])
            except RuntimeError:
                cache[video_id] = None
        return cache[video_id]

    unmeasured_rows = 0
    blocks = (np.flatnonzero(np.r_[True, prepared.clip_index[1:]
                                   != prepared.clip_index[:-1]])
              if len(prepared) else np.zeros(0, dtype=np.int64))
    bounds = np.r_[blocks, len(prepared)]
    for n_done, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        row = int(prepared.clip_index[a])
        source = waveform(sources[row])
        donor = waveform(donors[row]) if donors[row] else None
        if source is not None:
            spliced = spliced_waveform(source, corpus.row_segments(row),
                                       str(corpus.kinds[row]), donor)
            try:
                features[a:b] = extract_clip(
                    spliced, prepared.start_frame[a:b], prepared.end_frame[a:b],
                    require_edge=False).features
            except ValueError:
                source = None
        if source is None:
            unmeasured_rows += 1
        if n_done % 500 == 0:
            print(f"boundary cues: {n_done} corpus rows", flush=True)
    store = BoundaryCueStore(start_frame=prepared.start_frame,
                             end_frame=prepared.end_frame,
                             features=features.astype(np.float64),
                             column_names=column_names())
    output.parent.mkdir(parents=True, exist_ok=True)
    store.save(output)
    n_nan = int(np.isnan(features[:, column_names().index("audio_boundary_cue")]).sum())
    print(f"wrote {output}: {len(store)} proposal rows; {unmeasured_rows} corpus "
          f"rows without audio features; {n_nan} rows without an edge measurement")

    return output


def fit_proposal_ranker(proposals, corpus, boundary_cues, output, *,
                       features=None, protocol=None) -> Path:
    """Fit on self-splice assignments, retaining the adopted measurement mask.

    Feature deletions use the same measured rows as the three-feature fit.
    Additional cue features must also be measured on the retained rows.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    corpus_path = Path(corpus)
    corpus = SelfSpliceCorpus.load(corpus_path)
    run = load_emission_run(proposals, corpus.video_ids.tolist())
    prepared = prepare_proposals(run, RankingConfig())
    store = BoundaryCueStore.load(boundary_cues)
    features = FINAL_PROPOSAL_FEATURES if features is None else tuple(features)
    baseline = design_matrix(prepared, FINAL_PROPOSAL_FEATURES, store,
                             require_measured=False)
    X = (baseline if features == FINAL_PROPOSAL_FEATURES else
         design_matrix(prepared, features, store, require_measured=False))
    y = synthetic_segment_labels(prepared, corpus)
    measured = np.isfinite(baseline).all(axis=1) & np.isfinite(X).all(axis=1)
    ranker = ProposalRanker.fit(X[measured], y[measured], features)
    document = {
        "schema": "cast.proposal-ranker.v1", **ranker.to_dict(),
        "fit": {"rows": int(measured.sum()),
                "rows_excluded": int((~measured).sum()),
                "positive_assignments": int(y[measured].sum()),
                "assignment_iou": ASSIGNMENT_IOU, "ridge": RIDGE,
                "iterations": ITERATIONS,
                "measurement_features": list(dict.fromkeys(
                    (*FINAL_PROPOSAL_FEATURES, *features))),
                "exclusion_rule": "finite adopted-method and selected features; "
                                  "feature deletions retain the adopted measurement mask"},
        "labels": "self-splice synthetic segment bounds only",
        "benchmark_labels_used_for_fit": False,
        "boundary_curve": "full_band_flux_percentile", "pct_rule": "ecdf",
        "nofill": True,
        "self_splice_corpus": {"path": corpus_path.name},
    }
    if protocol:
        document["protocol"] = protocol
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: document[k] for k in ("w", "mu", "sd", "fit")}, indent=1))
    print(f"wrote {output}")
    return output


def _rank_proposals(proposals, clip_scores, labels, ranker, boundary_cues,
                    output, config) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    run = load_emission_run(proposals, clip_scores)
    run = reorder_to_labels(run, scored_clip_order(labels, run.video_ids))
    store = None if boundary_cues is None else BoundaryCueStore.load(boundary_cues)
    ranked = run_ranking_chain(run, ProposalRanker.load(ranker),
                               store, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    print(f"wrote {output}: {len(ranked)} proposals over "
          f"{ranked.video_id.nunique()} clips")
    return output


def rank_proposals(proposals, clip_scores, labels, ranker, boundary_cues, output) -> Path:
    """Write rankings; boundary_cues may be None for chain-only features."""
    return _rank_proposals(proposals, clip_scores, labels, ranker, boundary_cues,
                           output, RankingConfig())


__all__ = ["build_boundary_cues", "build_self_splice_boundary_cues",
           "fit_proposal_ranker", "rank_proposals"]
