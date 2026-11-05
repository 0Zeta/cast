"""Preprocess audio and video for the AV-HuBERT encoders."""


from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile
from skimage import transform as sk_transform

from cast.corpus import (DEPLOYED_PLACEMENT_LAW, filterbank_blend,
                         mouth_blend)
from cast.corpus.placement import ARTIFACT_PLACEMENT_LAW

STD_SIZE = (256, 256)
STABLE_POINTS = (33, 36, 39, 42, 45)
MOUTH_START, MOUTH_STOP = 48, 68
CROP_SIZE = 96
MODEL_CROP = 88
IMAGE_MEAN, IMAGE_STD = 0.421, 0.165
MAX_FRAMES = 750


def load_model(avhubert_root: str | Path, checkpoint: str | Path,
               device: str):
    """Load the pinned AV-HuBERT checkout without consuming this CLI's args."""
    module_dir = Path(avhubert_root).resolve() / "avhubert"
    if not (module_dir / "hubert.py").exists():
        raise FileNotFoundError(
            f"{module_dir} is not an AV-HuBERT checkout")
    sys.path.insert(0, str(module_dir))
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        fairseq = importlib.import_module("fairseq")
        for name in ("utils", "hubert_pretraining", "hubert",
                     "hubert_dataset", "hubert_asr", "hubert_criterion"):
            importlib.import_module(name)
    finally:
        sys.argv = saved_argv
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint)])
    model = models[0]
    if hasattr(model, "decoder"):
        model = model.encoder.w2v_model
    return model.to(device).eval()


def _interpolate_landmarks(landmarks):
    valid = [i for i, value in enumerate(landmarks) if value is not None]
    if not valid:
        return None
    for left, right in zip(valid[:-1], valid[1:]):
        if right - left == 1:
            continue
        delta = landmarks[right] - landmarks[left]
        for index in range(1, right - left):
            landmarks[left + index] = (
                landmarks[left] + index / float(right - left) * delta)
    landmarks[:valid[0]] = [landmarks[valid[0]]] * valid[0]
    landmarks[valid[-1]:] = [landmarks[valid[-1]]] * (
        len(landmarks) - valid[-1])
    return landmarks


def _detect_landmarks(frames, detector, cnn_detector, predictor):
    result = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = detector(gray, 1)
        if not rectangles:
            rectangles = [item.rect for item in cnn_detector(gray)]
        if not rectangles:
            result.append(None)
            continue
        shape = predictor(gray, rectangles[0])
        result.append(np.asarray([
            (shape.part(i).x, shape.part(i).y) for i in range(68)
        ], dtype=np.int32))
    return result


def _crop_mouth(image: np.ndarray, cx: float, cy: float) -> np.ndarray:
    half = CROP_SIZE // 2
    height, width = image.shape[:2]
    x0, x1 = max(0, int(round(cx - half))), min(width,
                                                       int(round(cx + half)))
    y0, y1 = max(0, int(round(cy - half))), min(height,
                                                        int(round(cy + half)))
    patch = image[y0:y1, x0:x1]
    if patch.shape[:2] != (CROP_SIZE, CROP_SIZE):
        patch = cv2.resize(patch, (CROP_SIZE, CROP_SIZE))
    return patch


def preprocess_video(path: str | Path, mean_face: np.ndarray, detector,
                     cnn_detector, predictor,
                     window_margin: int = 12,
                     max_frames: int | None = MAX_FRAMES):
    """Decode, align, and crop the 96x96 mouth sequence."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        return None, 0.0, 0

    landmarks = _detect_landmarks(
        frames, detector, cnn_detector, predictor)
    coverage = sum(item is not None for item in landmarks) / len(landmarks)
    landmarks = _interpolate_landmarks(landmarks)
    if landmarks is None:
        return None, coverage, len(frames)

    frame_queue, landmark_queue = deque(), deque()
    sequence = []
    margin = min(len(frames), window_margin)
    transform = None
    for frame, landmark in zip(frames, landmarks):
        frame_queue.append(frame)
        landmark_queue.append(landmark)
        if len(frame_queue) != margin:
            continue
        smoothed = np.mean(landmark_queue, axis=0)
        current_landmark = landmark_queue.popleft()
        current_frame = frame_queue.popleft()
        transform = sk_transform.estimate_transform(
            "similarity", smoothed[list(STABLE_POINTS), :],
            mean_face[list(STABLE_POINTS), :])
        warped = (sk_transform.warp(
            current_frame, inverse_map=transform.inverse,
            output_shape=STD_SIZE) * 255).astype(np.uint8)
        transformed = transform(current_landmark)
        center = np.mean(transformed[MOUTH_START:MOUTH_STOP], axis=0)
        sequence.append(_crop_mouth(warped, center[0], center[1]))
    while frame_queue:
        current_frame = frame_queue.popleft()
        current_landmark = landmark_queue.popleft()
        warped = (sk_transform.warp(
            current_frame, inverse_map=transform.inverse,
            output_shape=STD_SIZE) * 255).astype(np.uint8)
        transformed = transform(current_landmark)
        center = np.mean(transformed[MOUTH_START:MOUTH_STOP], axis=0)
        sequence.append(_crop_mouth(warped, center[0], center[1]))

    mouth = np.stack(sequence)
    original_length = len(mouth)
    if max_frames is not None:
        mouth = mouth[:max_frames]
    return mouth, coverage, original_length


def starting_silence(path: str | Path, threshold: float = 0.0005,
                     sample_rate: int = 16_000) -> tuple[float, int]:
    """Return the AVH-Align/AVPF audio onset and 25-Hz skip."""
    process = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
        "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ], capture_output=True, timeout=60)
    if process.returncode != 0:
        raise RuntimeError("audio_onset_decode_failed")
    waveform = np.frombuffer(process.stdout, dtype=np.float32)
    if not len(waveform):
        raise RuntimeError("no_audio_samples")
    loud = np.flatnonzero(np.abs(waveform) > threshold)
    silence_sec = float((loud[0] if len(loud) else len(waveform))
                        / sample_rate)
    skip_frames = int(silence_sec * 25) + 1 if silence_sec > 0 else 0
    return silence_sec, skip_frames


def preprocess_audio(path: str | Path, target_length: int,
                     skip_samples: int = 0) -> np.ndarray:
    """Decode mono 16-kHz audio and produce stacked 104-d log-fbanks."""
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)
    ], capture_output=True, text=True, timeout=30)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError("no_audio_stream")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = handle.name
    try:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(path),
            "-ac", "1", "-ar", "16000", "-f", "wav", wav_path
        ], check=True, timeout=60)
        sample_rate, waveform = wavfile.read(wav_path)
        if sample_rate != 16_000:
            raise RuntimeError(f"decoded sample rate is {sample_rate}")
        if skip_samples < 0:
            raise ValueError("skip_samples must be nonnegative")
        if skip_samples:
            waveform = waveform[skip_samples:]
        if not len(waveform):
            raise RuntimeError("empty_audio_after_trim")
        features = logfbank(waveform, samplerate=sample_rate).astype(np.float32)
        remainder = len(features) % 4
        if remainder:
            features = np.concatenate([
                features,
                np.zeros((4 - remainder, features.shape[1]),
                         dtype=features.dtype),
            ])
        features = features.reshape(-1, 4 * features.shape[1])
        if len(features) > target_length:
            features = features[:target_length]
        elif len(features) < target_length:
            features = np.concatenate([
                features,
                np.zeros((target_length - len(features), features.shape[1]),
                         dtype=features.dtype),
            ])
        return features
    finally:
        Path(wav_path).unlink(missing_ok=True)


def _model_inputs(mouth: np.ndarray, audio: np.ndarray, device: str):
    gray = np.stack([
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in mouth
    ])
    offset = (CROP_SIZE - MODEL_CROP) // 2
    gray = gray[:, offset:offset + MODEL_CROP,
                offset:offset + MODEL_CROP].astype(np.float32) / 255.0
    gray = (gray - IMAGE_MEAN) / IMAGE_STD
    video_tensor = (torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
                    .to(device))
    audio_tensor = torch.from_numpy(audio).to(device)
    audio_tensor = F.layer_norm(audio_tensor, audio_tensor.shape[-1:])
    audio_tensor = audio_tensor.transpose(0, 1).unsqueeze(0)
    return video_tensor, audio_tensor


@torch.no_grad()
def _forward(model, mouth: np.ndarray, audio: np.ndarray, device: str):
    video_tensor, audio_tensor = _model_inputs(mouth, audio, device)
    audio_out, _ = model.extract_finetune(
        source={"audio": audio_tensor, "video": None},
        padding_mask=None, mask=False)
    visual_out, _ = model.extract_finetune(
        source={"audio": None, "video": video_tensor},
        padding_mask=None, mask=False)
    joint_out, _ = model.extract_finetune(
        source={"audio": audio_tensor, "video": video_tensor},
        padding_mask=None, mask=False)
    cosine = (F.normalize(audio_out, dim=-1)
              * F.normalize(visual_out, dim=-1)).sum(-1).squeeze(0)
    return (
        audio_out.squeeze(0).to(torch.float16).cpu().numpy(),
        visual_out.squeeze(0).to(torch.float16).cpu().numpy(),
        joint_out.squeeze(0).to(torch.float16).cpu().numpy(),
        cosine.to(torch.float32).cpu().numpy(),
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _read_manifest(path: str | Path) -> list[tuple[str, Path]]:
    manifest = Path(path)
    rows = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("video_id", "")).strip()
            source = str(row.get("video_path", "")
                         or row.get("path", "") or video_id).strip()
            if not video_id or not source:
                continue
            source_path = Path(os.path.expandvars(source))
            if not source_path.is_absolute():
                source_path = manifest.parent / source_path
            rows.append((video_id, source_path.resolve()))
    if not rows:
        raise ValueError(f"{manifest} contains no video_id rows")
    ids = [video_id for video_id, _ in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{manifest} contains duplicate video_id values")
    return rows


def extract_manifest(
        manifest: str | Path, output_dir: str | Path,
        avhubert_root: str | Path, checkpoint: str | Path,
        cnn_detector: str | Path, face_predictor: str | Path,
        mean_face_path: str | Path, *, mode: str = "clean",
        rank: int = 0, nshard: int = 1, chunk_clips: int = 100,
        device: str = "cuda", log_every: int = 25,
        leading_silence_trim: bool = False,
        trim_order: str = "trim_then_cap",
        blend_purpose: str = "corpus") -> dict:
    """Extract one deterministic manifest shard into tar parts."""
    if mode not in {"clean", "visual-blend", "audio-blend"}:
        raise ValueError(f"unknown extraction mode {mode!r}")
    if blend_purpose not in {"corpus", "artifact-head"}:
        raise ValueError(f"unknown blend purpose {blend_purpose!r}")
    if trim_order not in {"cap_then_trim", "trim_then_cap"}:
        raise ValueError(f"unknown trim order {trim_order!r}")
    blend_law = (ARTIFACT_PLACEMENT_LAW if blend_purpose == "artifact-head"
                 else DEPLOYED_PLACEMENT_LAW)
    if leading_silence_trim and mode != "clean":
        raise ValueError(
            "leading-silence trim is an evaluation protocol and cannot "
            "be combined with self-splice modes")
    if not 0 <= rank < nshard:
        raise ValueError(f"rank {rank} must be in [0, {nshard})")
    if chunk_clips < 1:
        raise ValueError("chunk_clips must be positive")

    rows = _read_manifest(manifest)
    per_shard = math.ceil(len(rows) / nshard)
    assigned = rows[rank * per_shard:min(len(rows), (rank + 1) * per_shard)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"summary_shard_{rank:03d}.csv"
    existing = list(output.glob(f"shard_{rank:03d}_part_*.tar"))
    if summary_path.exists() or existing:
        raise FileExistsError(
            f"rank {rank} already has outputs in {output}; use a fresh "
            "directory or remove that rank's files explicitly")

    detector = dlib.get_frontal_face_detector()
    cnn = dlib.cnn_face_detection_model_v1(str(cnn_detector))
    predictor = dlib.shape_predictor(str(face_predictor))
    mean_face = np.load(mean_face_path, allow_pickle=False)
    model = load_model(avhubert_root, checkpoint, device)
    rng = np.random.default_rng(1000 + rank)
    previous_mouth = None
    previous_audio = None

    fields = ("video_id", "path", "ok", "reason", "T", "T_orig",
              "truncated", "coverage", "segments")
    if leading_silence_trim:
        fields += ("trimmed", "silence_sec", "skip_frames",
                   "T_posttrim_orig", "trim_cap_order", "media_time_origin_s")
    summary_rows = []
    archive = None
    part_index = -1
    ok_count = fail_count = 0
    started = time.time()
    try:
        for index, (video_id, source_path) in enumerate(assigned):
            if index % chunk_clips == 0:
                if archive is not None:
                    archive.close()
                part_index += 1
                archive = tarfile.open(
                    output / f"shard_{rank:03d}_part_{part_index:03d}.tar",
                    mode="w")
            result = {"video_id": video_id, "path": str(source_path),
                      "ok": False, "reason": "", "segments": []}
            try:
                silence_sec = 0.0
                skip_frames = 0
                trim_before_cap = (leading_silence_trim
                                   and trim_order == "trim_then_cap")
                if trim_before_cap:
                    silence_sec, skip_frames = starting_silence(source_path)
                mouth, coverage, original_length = preprocess_video(
                    source_path, mean_face, detector, cnn, predictor,
                    max_frames=None if trim_before_cap else MAX_FRAMES)
                result.update(coverage=float(coverage),
                              T_orig=int(original_length),
                              truncated=bool(original_length > MAX_FRAMES))
                if leading_silence_trim and not trim_before_cap:
                    # The original extractor checks the capped video before onset.
                    if mouth is None or len(mouth) < 5:
                        result["truncated"] = False
                        raise RuntimeError("video_preprocessing_failed")
                    silence_sec, skip_frames = starting_silence(source_path)
                posttrim_length = max(original_length - skip_frames, 0)
                if leading_silence_trim and mouth is not None:
                    mouth = np.ascontiguousarray(
                        mouth[skip_frames:skip_frames + MAX_FRAMES])
                if trim_before_cap:
                    result["truncated"] = bool(posttrim_length > MAX_FRAMES)
                if leading_silence_trim:
                    result.update(
                        trimmed=True,
                        silence_sec=float(silence_sec),
                        skip_frames=int(skip_frames),
                        T_posttrim_orig=int(posttrim_length),
                        trim_cap_order=f"{trim_order}_750",
                        media_time_origin_s=float(skip_frames / 25),
                    )
                if mouth is None or len(mouth) < 5:
                    reason = ("too_short_after_trim" if leading_silence_trim
                              and mouth is not None
                              else "video_preprocessing_failed")
                    raise RuntimeError(reason)
                if mode == "visual-blend":
                    clean_mouth = mouth.copy()
                    mouth, segments = mouth_blend(
                        mouth, previous_mouth, rng, blend_law)
                    previous_mouth = clean_mouth
                    result["segments"] = segments
                audio = preprocess_audio(
                    source_path, len(mouth),
                    skip_samples=skip_frames * 640)
                if mode == "audio-blend":
                    clean_audio = audio.copy()
                    audio, segments = filterbank_blend(
                        audio, previous_audio, rng, blend_law)
                    previous_audio = clean_audio
                    result["segments"] = segments
                feat_a, feat_v, feat_j, cosine = _forward(
                    model, mouth, audio, device)
                result.update(ok=True, T=int(len(cosine)))
                key = hashlib.sha1(video_id.encode()).hexdigest()
                metadata = dict(result)
                _add_bytes(archive, f"{key}.json",
                           json.dumps(metadata).encode())
                for suffix, array in (
                        ("audio.npy", feat_a), ("visual.npy", feat_v),
                        ("joint.npy", feat_j), ("cos.npy", cosine)):
                    _add_bytes(archive, f"{key}.{suffix}",
                               _npy_bytes(array))
                ok_count += 1
            except Exception as error:
                result["reason"] = str(error)[:200]
                fail_count += 1
            summary_rows.append(result)
            processed = index + 1
            if processed % log_every == 0 or processed == len(assigned):
                elapsed = (time.time() - started) / 60
                print(f"[{rank}/{nshard}] {processed}/{len(assigned)} "
                      f"ok={ok_count} failed={fail_count} "
                      f"elapsed={elapsed:.1f} min", flush=True)
    finally:
        if archive is not None:
            archive.close()

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    result = {"assigned": len(assigned), "ok": ok_count,
              "failed": fail_count, "rank": rank, "nshard": nshard,
              "mode": mode, "summary": str(summary_path)}
    if mode != "clean":
        result["blend_purpose"] = blend_purpose
        result["placement_law"] = {"mu": blend_law.mu, "sigma": blend_law.sigma,
                                   "max_frames": blend_law.max_frames}
    if leading_silence_trim:
        result["leading_silence_trim"] = True
        result["trim_order"] = trim_order
    return result
