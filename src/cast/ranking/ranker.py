"""Fit, load, and apply the cross-clip proposal ranker."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..artifacts import SchemaError, _require

FINAL_PROPOSAL_FEATURES: tuple[str, ...] = (
    "log1p_rank", "adjusted_score", "audio_boundary_cue")

RIDGE = 1e-4
ITERATIONS = 25
SD_FLOOR = 1e-9
STEP_TOLERANCE = 1e-7


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE,
                 iterations: int = ITERATIONS,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newton-IRLS logistic regression with an L2 ridge on standardized
    features and the intercept. Returns (weights, mean, sd)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _require(X.ndim == 2 and len(X) == len(y) and len(X) > 0,
             "the design matrix and labels must be nonempty and aligned")
    _require(bool(np.isfinite(X).all()), "non-finite value in the fit design")
    _require(set(np.unique(y).tolist()) == {0.0, 1.0},
             "the fit needs both positive and negative rows")
    mu, sd = X.mean(0), np.maximum(X.std(0), SD_FLOOR)
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Xs.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        gradient = Xs.T @ (p - y) + ridge * w
        curvature = p * (1 - p) + 1e-9
        hessian = (Xs * curvature[:, None]).T @ Xs + ridge * np.eye(Xs.shape[1])
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < STEP_TOLERANCE:
            break
    return w, mu, sd


@dataclass(frozen=True)
class ProposalRanker:
    """A logistic posterior over standardized proposal features."""

    weights: np.ndarray          # (k+1,), intercept first
    mean: np.ndarray             # (k,)
    sd: np.ndarray               # (k,)
    feature_names: tuple[str, ...]
    #: every other field of the artifact: scientific fit settings and protocol identity
    metadata: dict = field(default_factory=dict)

    KEYS = ("w", "mu", "sd", "features")

    @classmethod
    def load(cls, path: str | Path) -> "ProposalRanker":
        m = json.loads(Path(path).read_text())
        missing = [k for k in cls.KEYS if k not in m]
        _require(not missing, f"{path}: proposal ranker is missing keys {missing}")
        w = np.asarray(m["w"], dtype=np.float64)
        mu = np.asarray(m["mu"], dtype=np.float64)
        sd = np.asarray(m["sd"], dtype=np.float64)
        names = tuple(str(c) for c in m["features"])
        _require(len(names) > 0 and len(set(names)) == len(names),
                 f"{path}: features must be a nonempty list of distinct names")
        _require(len(w) == len(names) + 1,
                 f"{path}: {len(w)} weights do not fit {len(names)} features "
                 "plus an intercept")
        _require(len(mu) == len(names) and len(sd) == len(names),
                 f"{path}: standardization vectors do not match the feature count")
        _require(bool(np.isfinite(w).all() and np.isfinite(mu).all()
                      and np.isfinite(sd).all() and (sd > 0).all()),
                 f"{path}: non-finite or non-positive model constants")
        return cls(weights=w, mean=mu, sd=sd, feature_names=names,
                   metadata={k: v for k, v in m.items() if k not in cls.KEYS})

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
            ridge: float = RIDGE, iterations: int = ITERATIONS,
            ) -> "ProposalRanker":
        names = tuple(feature_names)
        _require(np.shape(X)[1] == len(names),
                 "the design matrix width does not match the feature names")
        w, mu, sd = fit_logistic(X, y, ridge, iterations)
        return cls(weights=w, mean=mu, sd=sd, feature_names=names)

    def to_dict(self) -> dict:
        return {"w": self.weights.tolist(), "mu": self.mean.tolist(),
                "sd": self.sd.tolist(), "features": list(self.feature_names)}

    def score(self, features: np.ndarray,
              feature_names: Sequence[str]) -> np.ndarray:
        """The standardized linear predictor; monotone in the posterior."""
        if tuple(feature_names) != self.feature_names:
            raise SchemaError(
                f"feature-name order {list(feature_names)} does not match "
                f"the fitted model's {list(self.feature_names)}")
        X = np.asarray(features, dtype=np.float64)
        _require(X.ndim == 2 and X.shape[1] == len(self.feature_names),
                 f"design matrix shape {X.shape} does not match "
                 f"{len(self.feature_names)} features")
        _require(bool(np.isfinite(X).all()),
                 "non-finite value in the ranker design matrix")
        post = np.c_[np.ones(len(X)), (X - self.mean) / self.sd] @ self.weights
        _require(bool(np.isfinite(post).all()), "non-finite ranker score")
        return post
