"""Train temporal artifact heads on frozen clean and self-blended features."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cast.extract.features import feature_parts
from cast.extract.heads import ArtifactHead, D_FEAT
from cast.training.data import FeatureCache


def _speaker(video_id: str) -> str:
    match = re.search(r"id\d+", video_id)
    return match[0] if match else video_id


def _load_blend(directory, stream):
    """Read the selected stream and its own annotations in TAR record order."""
    feature_suffix = f".{stream}.npy"
    segments_key = "blend_segs" if stream == "visual" else "blend_segs_a"
    rows = []
    for part in feature_parts(directory):
        pending = {}
        with tarfile.open(part) as archive:
            for member in archive:
                suffix = next((s for s in (feature_suffix, ".json")
                               if member.name.endswith(s)), None)
                if suffix is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                key = member.name[:-len(suffix)]
                row = pending.setdefault(key, {})
                row[suffix] = (json.load(handle) if suffix == ".json" else
                                np.load(io.BytesIO(handle.read()),
                                        allow_pickle=False).astype(np.float32))
        for row in pending.values():
            if feature_suffix not in row or ".json" not in row:
                continue
            metadata = row[".json"]
            rows.append((str(metadata["video_id"]), row[feature_suffix],
                         metadata.get(segments_key)
                         or metadata.get("segments") or []))
    return rows


def train_artifact_head(clean_features, blend_features, output, *,
                        stream="visual", epochs=8, learning_rate=3e-4,
                        seed=0, speaker_mod=0, speaker_keep=0,
                        speaker_salt="", dilated=False, boundary_aux=False,
                        aux_weight=0.3, device=None,
                        extra_negative_features=None) -> Path:
    """Save the best validation checkpoint for one visual or audio head.

    ``clean_features`` is a flat feature-cache prefix, a TAR directory, or a
    FeatureCache. Blend TARs carry inclusive ``blend_segs`` (visual) or
    ``blend_segs_a`` (audio), or the public extractor's ``segments`` field.
    Speaker filtering precedes the sorted speaker validation split. Extra
    negative TARs exclude validation speakers.

    Boundary supervision uses full sequences with masked losses. Otherwise,
    the input packs positive blended frames followed by sampled clean frames,
    matching the original artifact trainer.
    """
    if stream not in ("visual", "audio"):
        raise ValueError("stream must be 'visual' or 'audio'")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if speaker_mod < 0 or (speaker_mod and
                           not 0 <= speaker_keep < speaker_mod):
        raise ValueError("speaker_keep must be in [0, speaker_mod)")
    if not np.isfinite(aux_weight) or aux_weight < 0:
        raise ValueError("aux_weight must be finite and nonnegative")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stream_index = {"audio": 0, "visual": 1}[stream]
    blend = _load_blend(blend_features, stream)
    clean = (clean_features if isinstance(clean_features, FeatureCache)
             else FeatureCache(clean_features))
    clean_rows = {str(v): i for i, v in enumerate(clean.video_ids)}
    blend = [b for b in blend if b[0] in clean_rows and b[2]]
    if speaker_mod:
        blend = [b for b in blend if int(hashlib.sha1(
            (speaker_salt + _speaker(b[0])).encode()).hexdigest(), 16)
                 % speaker_mod == speaker_keep]
    if not blend:
        raise ValueError("no blended clips with segments and a clean counterpart "
                         "after speaker filtering")

    speakers = sorted({_speaker(b[0]) for b in blend})
    validation_speakers = (set(speakers[::10]) if len(speakers) >= 10
                           else set(speakers[:1]))
    validation_ids = {b[0] for b in blend
                      if _speaker(b[0]) in validation_speakers}
    extra_negative = []
    if extra_negative_features is not None:
        extra_negative = [(v, features, None) for v, features, _ in
                          _load_blend(extra_negative_features, stream)
                          if _speaker(v) not in validation_speakers]
    train = [b for b in blend if b[0] not in validation_ids] + extra_negative
    validation = [b for b in blend if b[0] in validation_ids]
    if not train:
        raise ValueError("no training clips after the speaker validation split")

    dimension = getattr(clean, stream).shape[1]
    for video_id, features, _ in blend + extra_negative:
        if (features.ndim != 2 or not len(features)
                or features.shape[1] != dimension):
            raise ValueError(f"invalid {stream} features for {video_id}: "
                             f"{features.shape}")

    def batches(items, training):
        order = list(range(len(items)))
        if training:
            rng.shuffle(order)
        for i in order:
            video_id, blended, segments = items[i]
            if segments is None:
                length = len(blended)
                negative_indices = rng.choice(length, min(length, 40),
                                               replace=False)
                yield (torch.from_numpy(blended[negative_indices].copy())[None],
                       torch.zeros(1, len(negative_indices)))
                continue
            original = np.asarray(clean.get3(clean_rows[video_id])[stream_index],
                                  dtype=np.float32)
            length = min(len(blended), len(original))
            positive_mask = np.zeros(length, dtype=bool)
            for start, end in segments:
                positive_mask[start:min(end + 1, length)] = True
            positive = blended[:length][positive_mask]
            negative_indices = rng.choice(
                length, min(length, max(20, 2 * len(positive))), replace=False)
            negative = original[:length][negative_indices]
            if not len(positive):
                continue
            x = np.concatenate([positive, negative])
            y = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
            yield torch.from_numpy(x)[None], torch.from_numpy(y)[None].float()

    def sequence_batches(items, training):
        order = list(range(len(items)))
        if training:
            rng.shuffle(order)
        for i in order:
            video_id, blended, segments = items[i]
            if segments is None:
                length = len(blended)
                negative_mask = np.zeros(length, dtype=bool)
                negative_mask[rng.choice(length, min(length, 40),
                                          replace=False)] = True
                yield (torch.from_numpy(blended.copy())[None],
                       torch.from_numpy(negative_mask), 0.0, None, None)
                continue
            original = np.asarray(clean.get3(clean_rows[video_id])[stream_index],
                                  dtype=np.float32)
            length = min(len(blended), len(original))
            main_mask = np.zeros(length, dtype=bool)
            distance = np.full(length, 99, dtype=np.int32)
            indices = np.arange(length)
            for start, end in segments:
                main_mask[start:min(end + 1, length)] = True
                for edge in (start, min(end, length - 1)):
                    distance = np.minimum(distance, np.abs(indices - edge))
            edge_mask = distance <= 3
            edge_target = (distance <= 1).astype(np.float32)
            if not main_mask.any() or not edge_mask.any():
                continue
            yield (torch.from_numpy(blended[:length].copy())[None],
                   torch.from_numpy(main_mask), None,
                   torch.from_numpy(edge_mask), torch.from_numpy(edge_target))
            negative_mask = np.zeros(length, dtype=bool)
            negative_mask[rng.choice(
                length, min(length, max(20, 2 * int(main_mask.sum()))),
                replace=False)] = True
            yield (torch.from_numpy(original[:length].copy())[None],
                   torch.from_numpy(negative_mask), 0.0, None, None)

    arch = {"dilated": dilated, "boundary_aux": boundary_aux}
    if dimension != D_FEAT:
        arch["d"] = dimension
    model = ArtifactHead(**arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    def run_epoch(items, training):
        total, count = 0.0, 0
        iterator = (sequence_batches if boundary_aux else batches)(items, training)
        for batch in iterator:
            if boundary_aux:
                x, main_mask, main_target, edge_mask, edge_target = batch
                logits, boundary = model.forward_both(x.to(device))
                mask = main_mask.to(device)
                target = (torch.ones if main_target is None else torch.zeros)(
                    int(mask.sum()), device=device)
                loss = loss_function(logits.squeeze(0)[mask], target)
                if edge_mask is not None:
                    mask = edge_mask.to(device)
                    loss = loss + aux_weight * loss_function(
                        boundary.squeeze(0)[mask], edge_target.to(device)[mask])
            else:
                x, y = batch
                loss = loss_function(model(x.to(device)), y.to(device))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.detach())
            count += 1
        if not count:
            raise ValueError("no supervised frames in "
                             + ("training" if training else "validation"))
        return total / count

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        training_loss = run_epoch(train, True)
        model.eval()
        with torch.no_grad():
            validation_loss = run_epoch(validation, False)
        print(f"epoch {epoch + 1}: train {training_loss:.4f} "
              f"val {validation_loss:.4f}", flush=True)
        if validation_loss < best:
            best = validation_loss
            torch.save({"state_dict": model.state_dict(), "arch": arch}, output)
    if not np.isfinite(best):
        raise ValueError("artifact training produced no finite validation loss")
    return output


def train_artifact_heads(clean_features, visual_blend_features,
                         audio_blend_features, output_dir, *, visual=None,
                         audio=None, crossfit=True, crossfit_salt="s5",
                         device=None) -> dict[str, Path]:
    """Train both main heads and, by default, their two speaker-fold twins.

    ``visual`` and ``audio`` supply per-stream training keyword arguments.
    Visual heads default to dilation and boundary supervision; audio heads
    default to neither. Crossfit role ``i`` trains on speaker parity ``i``;
    inference uses the complementary head for that speaker.
    """
    clean = (clean_features if isinstance(clean_features, FeatureCache)
             else FeatureCache(clean_features))
    output_dir = Path(output_dir)
    paths = {}
    for stream, features, options in (
            ("visual", visual_blend_features, visual),
            ("audio", audio_blend_features, audio)):
        parameters = {"dilated": stream == "visual",
                      "boundary_aux": stream == "visual"}
        parameters.update(options or {})
        parameters.update(stream=stream, device=device)
        role = f"{stream}_artifact_head"
        paths[role] = train_artifact_head(
            clean, features, output_dir / f"{role}.pt", **parameters)
        if crossfit:
            for fold in range(2):
                role = f"{stream}_artifact_crossfit_{fold}"
                fold_parameters = dict(parameters, speaker_mod=2,
                                       speaker_keep=fold,
                                       speaker_salt=crossfit_salt)
                paths[role] = train_artifact_head(
                    clean, features, output_dir / f"{role}.pt", **fold_parameters)
    return paths
