"""Run CAST score heads over extracted features."""


from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cast.fits.heads import head_checkpoint
from cast.speakers import speaker_hash_split

D_FEAT = 1024
K_LAG = 8
W_REF = 5
W_PRED = 5
D_PCA_PRED = 32


def _lag_mask(length: int, radius: int, device) -> torch.Tensor:
    t = torch.arange(length, device=device)[:, None]
    d = torch.arange(-radius, radius + 1, device=device)[None, :]
    index = t + d
    return (index >= 0) & (index < length)


class AlignmentHead(nn.Module):
    def __init__(self, d: int = D_FEAT, proj: int = 64,
                 radius: int = K_LAG):
        super().__init__()
        self.K = radius
        self.WA = nn.Linear(d, proj, bias=False)
        self.WV = nn.Linear(d, proj, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
        self.b = nn.Parameter(torch.zeros(2 * radius + 1))

    def logits(self, audio: torch.Tensor,
               visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.WV(visual), dim=-1)
        k = F.normalize(self.WA(audio), dim=-1)
        sims = []
        for lag in range(-self.K, self.K + 1):
            shifted = torch.roll(k, shifts=-lag, dims=1)
            sims.append((q * shifted).sum(-1))
        logits = (torch.stack(sims, dim=-1) / self.log_tau.exp()
                  + self.b)
        mask = _lag_mask(audio.shape[1], self.K, audio.device)[None]
        return logits.masked_fill(~mask, -1e9), mask

    def forward(self, audio: torch.Tensor,
                visual: torch.Tensor) -> torch.Tensor:
        logits, _ = self.logits(audio, visual)
        return logits.softmax(-1)


class ReferenceHead(nn.Module):
    def __init__(self, d: int = D_FEAT, hidden: int = 256,
                 radius: int = K_LAG, window: int = W_REF):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, kernel_size=2 * window + 1,
                      padding=window),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * radius + 1, kernel_size=1),
        )
        self.K = radius

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features.transpose(1, 2)).transpose(1, 2)
        mask = _lag_mask(features.shape[1], self.K,
                         features.device)[None]
        return logits.masked_fill(~mask, -1e9).softmax(-1)


class StudentTPredictor(nn.Module):
    def __init__(self, d_in: int = D_PCA_PRED,
                 d_out: int = D_PCA_PRED, window: int = W_PRED,
                 hidden: int = 512):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(d_in * (2 * window + 1), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * d_out),
        )
        self.nu_raw = nn.Parameter(torch.tensor(1.0))

    @property
    def nu(self):
        return 2.0 + F.softplus(self.nu_raw)

    def forward(self, context: torch.Tensor):
        output = self.net(context.flatten(-2))
        location, raw_scale = output.chunk(2, -1)
        return location, 0.01 + F.softplus(raw_scale)

    def nll(self, context: torch.Tensor, target: torch.Tensor):
        location, scale = self(context)
        nu = self.nu
        residual = (target - location) / scale
        log_normalizer = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
                          - 0.5 * torch.log(math.pi * nu))
        log_likelihood = (log_normalizer - torch.log(scale)
                          - (nu + 1) / 2 * torch.log1p(residual ** 2 / nu))
        return -log_likelihood.mean()


def _context(x: torch.Tensor, window: int = W_PRED) -> torch.Tensor:
    pieces = [torch.roll(x, shifts=-lag, dims=1)
              for lag in range(-window, window + 1)]
    context = torch.stack(pieces, dim=2)
    length = x.shape[1]
    for index, lag in enumerate(range(-window, window + 1)):
        if lag < 0:
            context[:, :-lag, index] = x[:, :1]
        elif lag > 0:
            context[:, length - lag:, index] = x[:, -1:]
    return context


def _student_t_scores(residual: torch.Tensor, nu: torch.Tensor):
    location = (nu + 1) * residual / (nu + residual ** 2)
    scale = -1.0 + (nu + 1) * residual ** 2 / (nu + residual ** 2)
    return location, scale


def _sync_residuals(pi: torch.Tensor, ref_a: torch.Tensor,
                    ref_v: torch.Tensor) -> torch.Tensor:
    root = pi.clamp_min(0).sqrt()
    return torch.cat([
        root - ref_a.clamp_min(0).sqrt(),
        root - ref_v.clamp_min(0).sqrt(),
    ], dim=-1)


class PredictionHeads:
    """Runtime alignment/reference heads and sync projection."""

    def __init__(self, checkpoint: str | Path, device: str):
        path = head_checkpoint("prediction_heads", checkpoint)
        saved = torch.load(path, map_location=device, weights_only=False)
        self.device = device
        self.align = AlignmentHead().to(device).eval()
        self.align.load_state_dict(saved["align"])
        self.ref_a = ReferenceHead().to(device).eval()
        self.ref_a.load_state_dict(saved["refA"])
        self.ref_v = ReferenceHead().to(device).eval()
        self.ref_v.load_state_dict(saved["refV"])
        self.a2v = StudentTPredictor().to(device).eval()
        self.a2v.load_state_dict(saved["a2v"])
        self.v2a = StudentTPredictor().to(device).eval()
        self.v2a.load_state_dict(saved["v2a"])
        pca = saved["sync_pca"]
        self.sync_mean = torch.from_numpy(pca["mean"]).to(device).float()
        self.sync_components = (torch.from_numpy(pca["comps"])
                                .to(device).float())
        prediction_pca = saved["pred_pca"]
        self.prediction_pca = {
            key: torch.from_numpy(value).to(device).float()
            for key, value in prediction_pca.items()
        }

    def _coordinates(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        components = (self.prediction_pca[f"{prefix}_comps"]
                      / self.prediction_pca[f"{prefix}_scale"][:, None])
        return ((x - self.prediction_pca[f"{prefix}_mean"])
                @ components.T)

    @torch.no_grad()
    def sync(self, audio: np.ndarray, visual: np.ndarray) -> np.ndarray:
        a = (torch.from_numpy(np.array(audio, copy=True, order="C"))
             .to(self.device).float()[None])
        v = (torch.from_numpy(np.array(visual, copy=True, order="C"))
             .to(self.device).float()[None])
        pi = self.align(a, v)
        residual = _sync_residuals(
            pi, self.ref_a(a), self.ref_v(v))[0]
        coords = ((residual - self.sync_mean)
                  @ self.sync_components.T)
        return coords.cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def scores(self, audio: np.ndarray,
               visual: np.ndarray) -> dict[str, np.ndarray]:
        a = (torch.from_numpy(np.array(audio, copy=True, order="C"))
             .to(self.device).float()[None])
        v = (torch.from_numpy(np.array(visual, copy=True, order="C"))
             .to(self.device).float()[None])
        pi = self.align(a, v)
        residual = _sync_residuals(
            pi, self.ref_a(a), self.ref_v(v))[0]
        output = {
            "sync": ((residual - self.sync_mean)
                     @ self.sync_components.T)
        }
        xa, xv = self._coordinates(a, "a"), self._coordinates(v, "v")
        for name, head, source, target in (
                ("a2v", self.a2v, xa, xv),
                ("v2a", self.v2a, xv, xa)):
            location, scale = head(_context(source))
            standardized = (target - location) / scale
            s_location, s_scale = _student_t_scores(
                standardized, head.nu)
            output[name] = torch.cat(
                [s_location, s_scale], dim=-1)[0]
        return {name: value.cpu().numpy().astype(np.float16)
                for name, value in output.items()}


class ArtifactHead(nn.Module):
    """Architecture stored by the visual/audio artifact checkpoints."""

    def __init__(self, d: int = D_FEAT, dilated: bool = False,
                 boundary_aux: bool = False):
        super().__init__()
        self.dilated = dilated
        if dilated:
            self.inp = nn.Sequential(
                nn.Conv1d(d, 256, 5, padding=2), nn.ReLU())
            self.mid_a = nn.Conv1d(256, 32, 3, padding=2, dilation=2)
            self.mid_b = nn.Conv1d(256, 32, 3, padding=4, dilation=4)
            self.out = nn.Conv1d(64, 1, 1)
        else:
            self.net = nn.Sequential(
                nn.Conv1d(d, 256, 5, padding=2), nn.ReLU(),
                nn.Conv1d(256, 64, 3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 1, 1),
            )
        self.out_bnd = nn.Conv1d(64, 1, 1) if boundary_aux else None

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        if self.dilated:
            h = self.inp(x.transpose(1, 2))
            return torch.relu(torch.cat([
                self.mid_a(h), self.mid_b(h)
            ], dim=1))
        return self.net[:4](x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.out if self.dilated else self.net[4]
        return head(self._trunk(x)).squeeze(1)

    def forward_both(self, x: torch.Tensor):
        features = self._trunk(x)
        head = self.out if self.dilated else self.net[4]
        boundary = (self.out_bnd(features).squeeze(1)
                    if self.out_bnd is not None else None)
        return head(features).squeeze(1), boundary


def load_artifact_head(name: str, checkpoint: str | Path,
                       device: str) -> ArtifactHead:
    path = head_checkpoint(name, checkpoint)
    saved = torch.load(path, map_location=device, weights_only=False)
    model = ArtifactHead(**saved.get("arch", {})).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


@torch.no_grad()
def artifact_scores(model: ArtifactHead, features: np.ndarray,
                    device: str) -> np.ndarray:
    x = (torch.from_numpy(np.ascontiguousarray(features,
                                               dtype=np.float32))
         .unsqueeze(0).to(device))
    return (-model(x).squeeze(0).cpu().numpy()).astype(np.float16)


class CrossFitArtifactHeads:
    """Artifact-head pairs used only to rebuild the self-splice corpus."""

    def __init__(self, visual: tuple[str | Path, str | Path],
                 audio: tuple[str | Path, str | Path], device: str,
                 salt: str = "s5"):
        self.device = device
        self.salt = salt
        self.visual = tuple(_load_crossfit_artifact(
            f"visual_artifact_crossfit_{i}", p, device)
                            for i, p in enumerate(visual))
        self.audio = tuple(_load_crossfit_artifact(
            f"audio_artifact_crossfit_{i}", p, device)
                           for i, p in enumerate(audio))

    def _head(self, pair, video_id: str):
        parity = speaker_hash_split(video_id, 2, self.salt)
        return pair[1 - parity]

    def scores(self, video_id: str, audio: np.ndarray,
               visual: np.ndarray) -> np.ndarray:
        v = artifact_scores(self._head(self.visual, video_id), visual,
                            self.device)
        a = artifact_scores(self._head(self.audio, video_id), audio,
                            self.device)
        return np.stack([v, a], axis=1)


def _load_crossfit_artifact(name: str, checkpoint: str | Path,
                            device: str) -> ArtifactHead:
    path = head_checkpoint(name, checkpoint)
    saved = torch.load(path, map_location=device, weights_only=False)
    model = ArtifactHead(**saved.get("arch", {})).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()
