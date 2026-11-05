"""Sample self-splice segment lengths and start frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True)
class PlacementLaw:
    mu: float
    sigma: float
    max_frames: int


# Exact literals the corpus generator ran with. Swapping in the four-decimal
# fit quotation changes the integer segment geometry and the corpus RNG stream.

DEPLOYED_PLACEMENT_LAW: Final[PlacementLaw] = PlacementLaw(
    mu=2.093, sigma=0.592, max_frames=33)

# Artifact heads use independently generated self-blends with this shorter law.
ARTIFACT_PLACEMENT_LAW: Final[PlacementLaw] = PlacementLaw(
    mu=1.95, sigma=0.5, max_frames=20)


def segment_length(rng: np.random.Generator, T: int,
                   law: PlacementLaw) -> int:
    hi = min(law.max_frames, T - 2)
    return int(np.clip(rng.lognormal(law.mu, law.sigma), 3, hi))


def segment(rng: np.random.Generator, T: int,
            law: PlacementLaw) -> tuple[int, int]:
    """Draw (start, length), consuming the length draw first."""
    L = segment_length(rng, T, law)
    return int(rng.integers(0, T - L)), L
