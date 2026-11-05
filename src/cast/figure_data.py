"""Select scientific figure inputs from a completed scan and ranking run.

No reference fitting, score standardization, or audio-boundary computation
belongs here. Those values are captured by the ordinary scientific stages.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import FPS, LabelTable, ProposalTable
from .eval.localization import proposal_seconds
from .eval.matching import iou_matrix
from .ranking.audio_boundary import EDGE_STEPS, SAMPLE_RATE, STFT_HOP

# Public gallery keys retain the paper's original score-set notation.
TRACK_NAMES = {
    "cross_modal_agreement": "cos",
    "audio_side_avsr_agreement": "ja_B",
    "video_side_avsr_agreement": "jv_B",
    "synchrony_residual": "sync",
    "visual_artifact_score": "vart_C",
    "audio_artifact_score": "aart_C",
    "fused_frame_score": "fused",
}
TRACK_LABELS = {
    "cos": "Cross-modal agreement", "ja_B": "Audio-side AVSR agreement",
    "jv_B": "Video-side AVSR agreement", "sync": "Synchrony residual",
    "vart_C": "Visual artifact score", "aart_C": "Audio artifact score",
    "fused": "Fused frame score", "flux": "Full-band flux percentile",
}


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_cases(path: str | Path) -> list[dict]:
    """Read a list, a ``cases`` list, or an object keyed by case ID."""
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        value = value.get("cases", value)
        if isinstance(value, dict):
            value = [{"id": key, **entry} for key, entry in value.items()]
    if not isinstance(value, list) or not value:
        raise ValueError("cases must be a nonempty list or object")
    ids = set()
    for entry in value:
        if not isinstance(entry, dict) or not entry.get("video_id"):
            raise ValueError("each case needs video_id")
        case_id = entry.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
            raise ValueError("each case needs a safe directory id")
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
    return value


def _transform(models, key: str) -> dict:
    name = TRACK_NAMES.get(key, key)
    result = {"label": TRACK_LABELS.get(name, name), "forgery_direction": "down",
              "kind": "scalar", "reference": 0.0,
              "transform": "(raw - mu) / sigma_w"}
    if key == "fused_frame_score":
        return {**result, "kind": "fused", "forgery_direction": "up",
                "reference": None,
                "transform": "frame_feature_matrix @ frame_w"}
    result["mu"] = models[f"{key}_mu"].tolist()
    if f"{key}_W" in models:
        result.update(kind="reduced_vector", reference=1.0, forgery_direction="up",
                      transform="mean(square((raw - mu) @ W.T), axis=1)")
        for field in ("W", "B", "psi", "q", "rho", "kappa"):
            result[field] = models[f"{key}_{field}"].tolist()
    else:
        for field in ("sigma_w", "psi", "q", "rho", "kappa"):
            result[field] = models[f"{key}_{field}"].tolist()
    return result


def _media(entry: dict, row: dict, manifest: Path) -> dict:
    selection = entry.get("media", {})
    if isinstance(selection, list):
        selection = {key: True for key in selection}
    if not isinstance(selection, dict) or set(selection) - {"video", "audio", "frame_times_s"}:
        raise ValueError("media must select video/audio and optional frame_times_s")
    result = {}
    for kind in ("video", "audio"):
        request = selection.get(kind, False)
        if not request:
            continue
        source = request if isinstance(request, str) else row.get(f"{kind}_path", row.get("video_path"))
        if not source or pd.isna(source):
            raise ValueError(f"{entry['id']}: requested {kind} has no media path")
        expanded = os.path.expanduser(os.path.expandvars(str(source)))
        if re.search(r"\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
            raise ValueError(f"{entry['id']}: unresolved media path variable: {source}")
        path = Path(expanded)
        result[kind] = str((manifest.parent / path).resolve() if not path.is_absolute() else path)
    if "frame_times_s" in selection:
        times = np.asarray(selection["frame_times_s"], dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError("media.frame_times_s must be finite seconds")
        result["frame_times_s"] = times.tolist()
    return result


def _modalities(entry: dict, label: dict) -> dict:
    if "modality_labels" in entry:
        result = entry["modality_labels"]
        if set(result) != {"audio", "video"} or any(type(v) is not bool for v in result.values()):
            raise ValueError("modality_labels needs audio/video booleans")
        return result
    kinds = {"real": (False, False), "audio_modified": (True, False),
             "visual_modified": (False, True), "both_modified": (True, True)}
    if label.get("modify_type") in kinds:
        audio, video = kinds[label["modify_type"]]
        return {"audio": audio, "video": video}
    # Unspecified modality labels must not turn into scientific genuine labels.
    return {"audio": None, "video": None}


def _frame_rates(run: Path, label_rows: pd.DataFrame) -> dict | None:
    """Use the saved localization protocol's exact emitted-T/duration rule."""
    protocol_path = run / "protocol.json"
    if not protocol_path.is_file():
        return None
    protocol = json.loads(protocol_path.read_text())
    localization = protocol.get("evaluation", {}).get("localization") or {}
    frame_rate = localization.get("frame_rate", "constant")
    if frame_rate == "constant":
        return None
    if frame_rate != "per-clip":
        raise ValueError("frame_rate must be 'constant' or 'per-clip'")
    duration_column = localization.get("duration_column", "duration")
    scores = pd.read_csv(run / "clip_scores.csv", usecols=["video_id", "n_frames"])
    joined = label_rows.reset_index().merge(scores, on="video_id", how="inner",
                                            validate="one_to_one")
    return dict(zip(joined.video_id, joined.n_frames / joined[duration_column]))


def _export_decoder(run: Path, data: Path, video_id: str, rate: float, n_frames: int) -> None:
    """Select saved decoder output without decoding or estimating it again."""
    for source, name, extra in (("proposals_learned.csv", "candidates.csv", "rank"),
                                ("decoded_learned.csv", "decoded.csv", "marginal")):
        source_path, output = run / source, data / name
        if not source_path.is_file():
            output.unlink(missing_ok=True)
            continue
        table = pd.read_csv(source_path, float_precision="round_trip")
        required = {"video_id", "start_frame", "end_frame", "score", extra}
        if not required <= set(table):
            raise ValueError(f"{source_path}: expected {sorted(required)}")
        table = table.loc[table.video_id == video_id].copy()
        bounds = table[["start_frame", "end_frame"]].to_numpy(dtype=float)
        if (not np.isfinite(bounds).all() or not (bounds == np.floor(bounds)).all()
                or (table.start_frame < 0).any() or (table.end_frame >= n_frames).any()
                or (table.end_frame < table.start_frame).any()):
            raise ValueError(f"{video_id}: invalid saved {name} bounds")
        if not np.isfinite(table.score).all():
            raise ValueError(f"{video_id}: invalid saved {name} scores")
        table["start_s"] = table.start_frame / rate
        table["end_s"] = (table.end_frame + 1) / rate
        table.rename(columns={"start_frame": "start_f", "end_frame": "end_f"}).to_csv(
            output, index=False, float_format="%.17g")
    frame_path = run / "frame_tracks.npz"
    output = data / "decoder_tracks.csv"
    output.unlink(missing_ok=True)
    if frame_path.is_file():
        with np.load(frame_path, allow_pickle=False) as tracks:
            ids = tracks["video_ids"].tolist()
            if video_id not in ids or "learned_marginal_inclusion" not in tracks:
                return
            index = ids.index(video_id)
            a, b = map(int, tracks["offsets"][index:index + 2])
            values = tracks["learned_marginal_inclusion"][a:b]
            if len(values) != n_frames or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError(f"{video_id}: invalid saved decoder marginal")
            pd.DataFrame({"frame": np.arange(n_frames), "time_s": np.arange(n_frames) / rate,
                          "learned_marginal_inclusion": values}).to_csv(
                output, index=False, float_format="%.17g")


def export_figure_data(run_dir, cases_path, manifest_path, labels_path) -> Path:
    """Return ``run_dir/figure_data`` after selecting whole-clip scientific data.

    Cases accept id, video_id, crop=[start_s,end_s], selected_tracks (gallery
    keys or scientific group names), and media={video: true, audio: true}.
    The source CSV needs video_id; optional video_path/audio_path are relative
    to that CSV, and media_time_origin_s locates score time zero in the media.
    Proposals are read from proposals_ranked.csv (the final ranking output).
    Frame bounds remain inclusive; seconds use the saved localization protocol's
    timebase, falling back to 25 Hz for direct scanner runs. Boundary-curve
    times retain their canonical waveform clock.
    """
    run, manifest_path = Path(run_dir).resolve(), Path(manifest_path).resolve()
    cases = read_cases(cases_path)
    manifest = pd.read_csv(manifest_path)
    if "video_id" not in manifest or not manifest.video_id.is_unique:
        raise ValueError("source manifest needs unique video_id rows")
    manifest = manifest.set_index("video_id")
    label_rows = pd.read_csv(labels_path)
    labels = LabelTable.from_dataframe(label_rows)
    label_rows = label_rows.set_index("video_id")
    frame_rates = _frame_rates(run, label_rows)
    truth = dict(zip(labels.video_id, labels.segments))
    summary = json.loads((run / "fitted_models.json").read_text())
    fitted = run / summary.get("fitted_dir", "fitted")
    ranked_path = run / "proposals_ranked.csv"
    if not ranked_path.is_file():
        raise FileNotFoundError(f"final ranking output is required: {ranked_path}")
    proposals = pd.read_csv(ranked_path)
    ProposalTable.from_dataframe(proposals)
    if "rank" not in proposals:
        raise ValueError("final ranked proposals need a rank column")
    output = run / "figure_data"
    records = []
    with np.load(run / "scientific_tracks.npz", allow_pickle=False) as tracks:
        track_ids = tracks["video_ids"].tolist()
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("scientific tracks contain duplicate video_id")
        for entry in cases:
            video_id = entry["video_id"]
            if video_id not in track_ids:
                raise ValueError(f"{video_id}: no captured scientific tracks; request figure_video_ids during scan")
            if video_id not in manifest.index or video_id not in truth:
                raise ValueError(f"{video_id}: absent from source manifest or labels")
            index = track_ids.index(video_id)
            a, b = map(int, tracks["offsets"][index:index + 2])
            round_id = int(tracks["rounds"][index])
            row = manifest.loc[video_id].to_dict()
            origin = float(row.get("media_time_origin_s", 0.0))
            if not np.isfinite(origin) or origin < 0:
                raise ValueError("media_time_origin_s must be finite and nonnegative")
            rate = FPS if frame_rates is None else frame_rates.get(video_id)
            if rate is None or not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"{video_id}: clip frame rate must be finite and positive")
            duration = (b - a) / rate
            crop = entry.get("crop", [0.0, duration])
            if len(crop) != 2 or not (0 <= float(crop[0]) < float(crop[1]) <= duration):
                raise ValueError(f"{entry['id']}: crop must lie inside the scored clip")
            scientific = {TRACK_NAMES.get(key, key): key for key in tracks.files
                          if key not in ("video_ids", "offsets", "rounds")}
            selected = [TRACK_NAMES.get(key, key) for key in entry.get("selected_tracks", list(scientific))]
            if not selected or len(set(selected)) != len(selected) or set(selected) - (set(scientific) | {"flux"}):
                raise ValueError(f"{entry['id']}: invalid or absent selected_tracks")
            data = output / entry["id"] / "data"
            data.mkdir(parents=True, exist_ok=True)
            _export_decoder(run, data, video_id, rate, b - a)
            table = {"frame": np.arange(b - a), "time_s": np.arange(b - a) / rate}
            table.update({name: tracks[key][a:b] for name, key in scientific.items()})
            if not all(np.isfinite(value).all() for value in table.values()):
                raise ValueError(f"{entry['id']}: non-finite scientific tracks")
            pd.DataFrame(table).to_csv(data / "tracks.csv", index=False, float_format="%.17g")
            channel_path = fitted / f"round{round_id}_models.npz"
            weight_path = fitted / f"round{round_id}_weights.npz"
            with np.load(channel_path, allow_pickle=False) as models:
                channels = {name: _transform(models, key) for name, key in scientific.items()}
                scalar_order = [name for name, key in scientific.items()
                                if f"{key}_sigma_w" in models]
                vector_order = [name for name, key in scientific.items()
                                if f"{key}_W" in models]
            shutil.copyfile(channel_path, data / "fitted_models.npz")
            shutil.copyfile(weight_path, data / "fusion_weights.npz")
            gt = truth[video_id]
            p = proposals.loc[proposals.video_id == video_id].sort_values("rank", kind="stable").copy()
            if p["rank"].tolist() != list(range(len(p))):
                raise ValueError(f"{video_id}: final ranks must be contiguous from zero")
            p["start_s"], p["end_s"] = p.start_frame / rate, (p.end_frame + 1) / rate
            if len(p) and (p.end_frame >= b - a).any():
                raise ValueError(f"{video_id}: proposal outside captured frames")
            start_s, end_s = proposal_seconds(p.video_id.to_numpy(),
                                              p.start_frame.to_numpy(),
                                              p.end_frame.to_numpy(), frame_rates)
            pairwise = iou_matrix(start_s, end_s, gt[:, 0], gt[:, 1])
            p["iou"] = pairwise.max(axis=1) if len(gt) else np.zeros(len(p))
            p = p.rename(columns={"start_frame": "start_f", "end_frame": "end_f"})
            p.to_csv(data / "proposals.csv", index=False, float_format="%.17g")
            boundary = None
            for name in ("boundary_curve.csv", "boundary_peaks.csv"):
                (data / name).unlink(missing_ok=True)
            for path in sorted((run / "boundary_curves").glob("*.npz")):
                with np.load(path, allow_pickle=False) as curve:
                    if str(curve["video_id"].item()) != video_id:
                        continue
                    times = curve["time_seconds"]
                    values = curve["full_band_flux_percentile"]
                    if times.ndim != 1 or times.shape != values.shape or not len(times) or not np.isfinite(times).all() or not np.isfinite(values).all() or not (np.diff(times) > 0).all() or (values < 0).any() or (values > 1).any():
                        raise ValueError(f"invalid boundary curve: {path}")
                    pd.DataFrame({"time_s": times, "flux": values}).to_csv(
                        data / "boundary_curve.csv", index=False, float_format="%.17g")
                    peaks = []
                    if len(p):
                        proposal = p.iloc[0]
                        for edge, frame in (("start", proposal["start_f"]),
                                            ("end", proposal["end_f"] + 1)):
                            # Use the paper's frame-aligned display boundaries.
                            center = int(round(int(frame) * SAMPLE_RATE / (rate * STFT_HOP)))
                            lo, hi = max(0, center - EDGE_STEPS), min(len(values), center + EDGE_STEPS + 1)
                            if lo >= hi:
                                continue
                            peak = lo + int(np.argmax(values[lo:hi]))
                            peaks.append({"rank": int(proposal["rank"]), "edge": edge,
                                          "time_s": times[peak], "flux": values[peak]})
                    pd.DataFrame(peaks, columns=["rank", "edge", "time_s", "flux"]).to_csv(
                        data / "boundary_peaks.csv", index=False, float_format="%.17g")
                    boundary = "boundary_curve.csv"
                    channels["flux"] = {"label": TRACK_LABELS["flux"], "kind": "ranking_cue",
                                        "transform": "canonical full-band flux within-clip percentile",
                                        "forgery_direction": "up", "reference": None}
                    break
            if "flux" in selected and boundary is None:
                raise FileNotFoundError(f"{video_id}: requested flux needs captured boundary_curves")
            case = {"schema_version": 1, "id": entry["id"], "video_id": video_id,
                    "logical_video_id": video_id, "round": round_id, "fps": rate,
                    "n_frames": b - a, "duration_s": duration,
                    "media_time_origin_s": origin, "crop": list(map(float, crop)),
                    "selected_tracks": selected, "score_groups": list(scientific),
                    "interval_semantics": "inclusive frames; half-open seconds",
                    "ground_truth_intervals": [{"start_s": float(lo), "end_s": float(hi)} for lo, hi in gt],
                    "modality_labels": _modalities(entry, label_rows.loc[video_id].to_dict()),
                    "media": _media(entry, row, manifest_path), "boundary_curve": boundary,
                    "fitted_parameters": "fitted_models.npz", "fusion_weights": "fusion_weights.npz"}
            if isinstance(row.get("source_video_id"), str):
                case["source_video_id"] = row["source_video_id"]
            _json(data / "selected_case.json", case)
            _json(data / "reference_display_transform.json", {
                "round": round_id, "channels": channels,
                "frame_feature_order": scalar_order + vector_order,
                "frame_feature_signs": [-1] * len(scalar_order) + [1] * len(vector_order)})
            records.append({"id": entry["id"], "video_id": video_id, "round": round_id,
                            "data": f"{entry['id']}/data"})
    _json(output / "cases.json", {"schema_version": 1, "cases": records})
    return output
