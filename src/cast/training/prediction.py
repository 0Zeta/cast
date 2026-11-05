"""Train alignment, reference, PCA, and Student-t prediction heads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cast.extract.heads import (
    D_FEAT,
    D_PCA_PRED,
    K_LAG,
    AlignmentHead,
    ReferenceHead,
    StudentTPredictor,
    _context as make_context,
    _sync_residuals as sync_residuals,
)

from .data import FeatureCache


class WindowSampler:
    """Sample clips with replacement, then a uniform window within each clip."""

    def __init__(self, cache, idx, T_w=160, preload=True):
        self.T_w = T_w
        self.idx = [i for i in idx if cache.lengths[i] >= T_w]
        if not self.idx:
            raise ValueError(
                f"no clips with at least {T_w} frames in this speaker split")
        self.offsets = cache.offsets
        self.lengths = cache.lengths
        if preload:
            self.audio = np.asarray(cache.audio)
            self.visual = np.asarray(cache.visual)
        else:
            self.audio, self.visual = cache.audio, cache.visual

    def batch(self, bs, rng, device):
        rows = rng.choice(self.idx, bs)
        audio = np.empty((bs, self.T_w, D_FEAT), np.float16)
        visual = np.empty((bs, self.T_w, D_FEAT), np.float16)
        for j, i in enumerate(rows):
            start = rng.integers(0, self.lengths[i] - self.T_w + 1)
            offset = self.offsets[i] + start
            audio[j] = self.audio[offset:offset + self.T_w]
            visual[j] = self.visual[offset:offset + self.T_w]
        return (torch.from_numpy(audio).to(device).float(),
                torch.from_numpy(visual).to(device).float())


def hellinger_sq(p, q, eps=1e-8):
    # The epsilon keeps masked zero probabilities differentiable.
    return 0.5 * (((p + eps).sqrt() - (q + eps).sqrt()) ** 2).sum(-1)


def make_lag_field(B, T, K, device, p_constant=0.5):
    """Draw the original mixture of constant and smoothed synthetic lags."""
    constant = torch.randint(-K, K + 1, (B, 1), device=device).float().expand(B, T)
    noise = torch.randn(B, 1, T, device=device)
    smooth = F.avg_pool1d(F.pad(noise, (12, 12), mode="replicate"),
                          kernel_size=25, stride=1).squeeze(1)
    smooth = smooth / smooth.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) * K
    pick = (torch.rand(B, 1, device=device) < p_constant).float()
    return (pick * constant + (1 - pick) * smooth).round().long().clamp(-K, K)


def shift_stream(features, lag_field):
    """Gather features at t + lag, clamping to the clip edges."""
    batch, length, dimension = features.shape
    index = (torch.arange(length, device=features.device)[None]
             + lag_field).clamp(0, length - 1)
    return torch.gather(features, 1, index[..., None].expand(batch, length, dimension))


def shift_targets(lag_field, radius, mask):
    """Mask and normalize targets with weights 1 at -lag and .25 beside it."""
    batch, length = lag_field.shape
    centers = (-lag_field).clamp(-radius, radius) + radius
    target = torch.zeros(batch, length, 2 * radius + 1, device=lag_field.device)
    target.scatter_(-1, centers[..., None], 1.0)
    for offset, weight in ((-1, 0.25), (1, 0.25)):
        neighbor = (centers + offset).clamp(0, 2 * radius)
        target.scatter_add_(-1, neighbor[..., None],
                            torch.full_like(target[..., :1], weight))
    target = target * mask
    return target / target.sum(-1, keepdim=True).clamp_min(1e-9)


def fit_pca(values, d_max):
    mean = values.mean(0)
    _, _, components = np.linalg.svd(values - mean, full_matrices=False)
    return mean, components[:d_max]


def ppca_loglik(values, mean, components, dimension):
    """Use the golden held-out PPCA rank-selection criterion."""
    centered = values - mean
    projected = centered @ components[:dimension].T
    residual = centered - projected @ components[:dimension]
    width = values.shape[1]
    residual_variance = max(float(residual.var()), 1e-9)
    projected_variance = projected.var(axis=0)
    likelihood = -0.5 * (
        np.sum(np.log(2 * np.pi * np.maximum(projected_variance, 1e-9)))
        + dimension
        + (width - dimension) * np.log(2 * np.pi * residual_variance)
        + (width - dimension))
    return likelihood * len(values)


def train_prediction_heads(
    features,
    output,
    *,
    steps_align=3000,
    steps_ref=3000,
    steps_pred=5000,
    batch_size=48,
    learning_rate=3e-4,
    seed=0,
    device=None,
    window_frames=160,
    preload=True,
    augment=False,
    validation_batches=20,
    pca_train_batches=60,
    pca_validation_batches=15,
    prediction_validation_batches=15,
) -> Path:
    """Fit genuine-only heads and save a PredictionHeads-compatible checkpoint.

    Speaker SHA1 modulo ten reserves fold zero for validation. Stages share
    one NumPy RNG, with model initialization and augmentation using Torch's
    seeded RNG in the order of the original train_cast_heads.py script.
    """
    if any(steps < 0 for steps in (steps_align, steps_ref, steps_pred)):
        raise ValueError("training step counts must be nonnegative")
    if any(count <= 0 for count in (
            batch_size, window_frames, validation_batches, pca_train_batches,
            pca_validation_batches, prediction_validation_batches)):
        raise ValueError("batch size, window length, and batch counts must be positive")
    if (batch_size * window_frames * pca_train_batches + 1) // 2 < D_PCA_PRED:
        raise ValueError(f"predictive PCA needs at least {D_PCA_PRED} sampled frames "
                         "after subsampling every other frame")

    def maybe_augment(audio, visual):
        if not augment:
            return audio, visual
        batch = audio.shape[0]
        pick = torch.rand(batch, 1, 1, device=audio.device)
        sigma = torch.rand(batch, 1, 1, device=audio.device) * 0.15 + 0.05
        audio = torch.where(
            pick < 0.4, audio + sigma * audio.std() * torch.randn_like(audio), audio)
        pick_visual = torch.rand(batch, 1, 1, device=audio.device)
        visual = torch.where(
            pick_visual < 0.4,
            visual + sigma * visual.std() * torch.randn_like(visual), visual)
        return audio, visual

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cache = features if isinstance(features, FeatureCache) else FeatureCache(features)

    def speaker(video_id):
        match = re.search(r"/(id\d+)/", video_id)
        return match.group(1) if match else video_id

    fold = np.array([
        int(hashlib.sha1(speaker(video_id).encode()).hexdigest(), 16) % 10
        for video_id in cache.video_ids])
    train_indices = np.where(fold != 0)[0]
    validation_indices = np.where(fold == 0)[0]
    print(f"clips: {len(train_indices)} train / {len(validation_indices)} val", flush=True)
    train = WindowSampler(cache, train_indices, T_w=window_frames, preload=preload)
    validation = WindowSampler(cache, validation_indices, T_w=window_frames, preload=False)
    metrics = {}

    align = AlignmentHead().to(device)
    optimizer = torch.optim.AdamW(align.parameters(), lr=learning_rate)
    for step in range(1, steps_align + 1):
        audio, visual = train.batch(batch_size, rng, device)
        audio, visual = maybe_augment(audio, visual)
        lag_field = make_lag_field(batch_size, audio.shape[1], K_LAG, device)
        logits, mask = align.logits(shift_stream(audio, lag_field), visual)
        target = shift_targets(lag_field, K_LAG, mask)
        loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"[align {step}] loss={loss.item():.4f}", flush=True)
    align.eval()
    with torch.no_grad():
        accuracies = []
        for _ in range(validation_batches):
            audio, visual = validation.batch(batch_size, rng, device)
            lag_field = torch.randint(-K_LAG, K_LAG + 1, (batch_size, 1),
                                      device=device).expand(batch_size, audio.shape[1])
            pi = align(shift_stream(audio, lag_field), visual)
            predicted = pi.argmax(-1) - K_LAG
            accuracies.append((predicted == -lag_field).float().mean().item())
    metrics["align_val_lag_acc"] = float(np.mean(accuracies))

    ref_a, ref_v = ReferenceHead().to(device), ReferenceHead().to(device)
    optimizer = torch.optim.AdamW([*ref_a.parameters(), *ref_v.parameters()],
                                  lr=learning_rate)
    for step in range(1, steps_ref + 1):
        audio, visual = train.batch(batch_size, rng, device)
        audio, visual = maybe_augment(audio, visual)
        with torch.no_grad():
            pi = align(audio, visual)
        pa, pv = ref_a(audio), ref_v(visual)
        loss = (hellinger_sq(pi, pa) + hellinger_sq(pi, pv)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"[ref {step}] H2={loss.item():.4f}", flush=True)
    ref_a.eval()
    ref_v.eval()
    with torch.no_grad():
        distances = []
        for _ in range(validation_batches):
            audio, visual = validation.batch(batch_size, rng, device)
            pi = align(audio, visual)
            distances.append((hellinger_sq(pi, ref_a(audio))
                              + hellinger_sq(pi, ref_v(visual))).mean().item())
    metrics["ref_val_h2"] = float(np.mean(distances))

    with torch.no_grad():
        residuals = []
        for _ in range(pca_train_batches):
            audio, visual = train.batch(batch_size, rng, device)
            pi = align(audio, visual)
            residual = sync_residuals(pi, ref_a(audio), ref_v(visual))
            residuals.append(residual.reshape(-1, residual.shape[-1]).cpu().numpy())
        sync_train = np.concatenate(residuals)
        residuals = []
        for _ in range(pca_validation_batches):
            audio, visual = validation.batch(batch_size, rng, device)
            pi = align(audio, visual)
            residual = sync_residuals(pi, ref_a(audio), ref_v(visual))
            residuals.append(residual.reshape(-1, residual.shape[-1]).cpu().numpy())
        sync_validation = np.concatenate(residuals)
    sync_mean, sync_components = fit_pca(sync_train.astype(np.float64), 16)
    likelihoods = {
        d: ppca_loglik(sync_validation.astype(np.float64), sync_mean, sync_components, d)
        for d in (4, 6, 8, 12)}
    d_sync = max(likelihoods, key=likelihoods.get)
    metrics["d_sync"] = d_sync

    frames_a, frames_v = [], []
    for _ in range(pca_train_batches):
        audio, visual = train.batch(batch_size, rng, device)
        frames_a.append(audio.reshape(-1, D_FEAT).cpu().numpy())
        frames_v.append(visual.reshape(-1, D_FEAT).cpu().numpy())
    audio_frames = np.concatenate(frames_a)[::2].astype(np.float64)
    visual_frames = np.concatenate(frames_v)[::2].astype(np.float64)
    a_mean, a_components = fit_pca(audio_frames, D_PCA_PRED)
    v_mean, v_components = fit_pca(visual_frames, D_PCA_PRED)
    a_scale = np.std((audio_frames - a_mean) @ a_components.T, axis=0) + 1e-6
    v_scale = np.std((visual_frames - v_mean) @ v_components.T, axis=0) + 1e-6
    del audio_frames, visual_frames, frames_a, frames_v

    def to_coordinates(values, mean, components, scale):
        weights = torch.from_numpy((components / scale[:, None]).T).to(values.device).float()
        center = torch.from_numpy(mean).to(values.device).float()
        return (values - center) @ weights

    predictors = {}
    for name in ("a2v", "v2a"):
        head = StudentTPredictor().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
        for step in range(1, steps_pred + 1):
            audio, visual = train.batch(batch_size, rng, device)
            audio, visual = maybe_augment(audio, visual)
            xa = to_coordinates(audio, a_mean, a_components, a_scale)
            xv = to_coordinates(visual, v_mean, v_components, v_scale)
            source, target = (xa, xv) if name == "a2v" else (xv, xa)
            loss = head.nll(make_context(source), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                print(f"[{name} {step}] nll={loss.item():.4f} nu={head.nu.item():.2f}",
                      flush=True)
        head.eval()
        with torch.no_grad():
            losses = []
            for _ in range(prediction_validation_batches):
                audio, visual = validation.batch(batch_size, rng, device)
                xa = to_coordinates(audio, a_mean, a_components, a_scale)
                xv = to_coordinates(visual, v_mean, v_components, v_scale)
                source, target = (xa, xv) if name == "a2v" else (xv, xa)
                losses.append(head.nll(make_context(source), target).item())
        metrics[f"{name}_val_nll"] = float(np.mean(losses))
        predictors[name] = head

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "align": align.state_dict(),
        "refA": ref_a.state_dict(),
        "refV": ref_v.state_dict(),
        "a2v": predictors["a2v"].state_dict(),
        "v2a": predictors["v2a"].state_dict(),
        "sync_pca": {"mean": sync_mean, "comps": sync_components[:d_sync]},
        "pred_pca": {"a_mean": a_mean, "a_comps": a_components, "a_scale": a_scale,
                     "v_mean": v_mean, "v_comps": v_components, "v_scale": v_scale},
        "config": {"K": K_LAG, "d_sync": int(d_sync), "d_pred": D_PCA_PRED},
        "metrics": metrics,
    }, output)
    print(f"saved {output}\nmetrics: {json.dumps(metrics, indent=2)}", flush=True)
    return output
