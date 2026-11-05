"""Apply the four self-splice corruption operators."""

from __future__ import annotations

import numpy as np

from .placement import PlacementLaw, segment, segment_length

# ``a_shift`` uses a dedicated generator for its non-zero offset. Consume the
# retired shift draw from ``rng`` to preserve the pinned corpus RNG sequence.
AUDIO_SHIFT_SEED = 20260824
_RETIRED_AUDIO_SHIFT_FRAMES = (-8, -6, -4, -3, 3, 4, 6, 8)

# Per audio-local row, u ~ U(0, 1) selects a_replace iff u < this.
AUDIO_REPLACE_FRACTION = 0.6

Segments = list[tuple[int, int]]


def _match_statistics(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return ((src - src.mean()) / max(src.std(), 1e-3)
            * tgt.std() + tgt.mean())


def elliptical_mouth_mask(h: int, w: int, ndim: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Soft elliptical mask over the mouth region; geometry and feather
    randomized per segment so a head cannot latch onto a fixed blend
    footprint."""
    cy = float(rng.uniform(0.45, 0.65))
    cx = float(rng.uniform(0.42, 0.58))
    ry = float(rng.uniform(0.25, 0.48))
    rx = float(rng.uniform(0.3, 0.52))
    k = float(rng.uniform(1.15, 2.0))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (((yy - h * cy) / (h * ry)) ** 2 + ((xx - w * cx) / (w * rx)) ** 2)
    m = np.clip(k - d, 0.0, 1.0)
    return m[..., None] if ndim == 4 else m


def mouth_blend(mouth: np.ndarray, donor: np.ndarray | None,
                rng: np.random.Generator, law: PlacementLaw,
                ) -> tuple[np.ndarray, Segments]:
    """Blend one or two mouth segments and return inclusive bounds.

    Shape mismatches skip output but still consume RNG draws."""
    out = mouth.astype(np.float32).copy()
    T = len(out)
    segs: Segments = []
    for _ in range(int(rng.integers(1, 3))):
        L = segment_length(rng, T, law)
        s = int(rng.integers(0, T - L))
        alpha = float(rng.uniform(0.6, 1.0))
        mask = elliptical_mouth_mask(out.shape[1], out.shape[2],
                                     out.ndim, rng)
        if donor is not None and len(donor) >= L and rng.random() < 0.7:
            o = int(rng.integers(0, len(donor) - L + 1))
            src = donor[o:o + L].astype(np.float32)
        else:  # identity-preserving: own frames time-shifted
            d = int(rng.integers(15, max(16, T - L)))
            idx = (np.arange(s, s + L) + d) % T
            src = out[idx].copy()
        if src.shape[1:] != out.shape[1:]:
            continue
        tgt = out[s:s + L]
        src = _match_statistics(src, tgt)
        m = alpha * mask
        out[s:s + L] = (1 - m) * tgt + m * src
        segs.append((s, s + L - 1))
    return np.clip(out, 0, 255).astype(mouth.dtype), segs


def filterbank_blend(features: np.ndarray, donor: np.ndarray | None,
                     rng: np.random.Generator, law: PlacementLaw,
                     ) -> tuple[np.ndarray, Segments]:
    """Blend one or two statistic-matched filterbank segments with one-frame boundary crossfades."""
    out = features.astype(np.float32).copy()
    T = len(out)
    segs: Segments = []
    for _ in range(int(rng.integers(1, 3))):
        L = segment_length(rng, T, law)
        s = int(rng.integers(0, T - L))
        alpha = float(rng.uniform(0.7, 1.0))
        if donor is not None and len(donor) >= L and rng.random() < 0.7:
            o = int(rng.integers(0, len(donor) - L + 1))
            src = donor[o:o + L].astype(np.float32)
        else:  # own frames, time-shifted
            d = int(rng.integers(15, max(16, T - L)))
            idx = (np.arange(s, s + L) + d) % T
            src = out[idx].copy()
        tgt = out[s:s + L]
        src = _match_statistics(src, tgt)
        w = np.full(L, alpha, dtype=np.float32)
        w[0] *= 0.5
        w[-1] *= 0.5
        out[s:s + L] = (1 - w[:, None]) * tgt + w[:, None] * src
        segs.append((s, s + L - 1))
    return out.astype(features.dtype), segs


def audio_shift_generator() -> np.random.Generator:
    """The generator a_shift magnitudes are drawn from, kept separate
    from the build RNG (see AUDIO_SHIFT_SEED)."""
    return np.random.default_rng(AUDIO_SHIFT_SEED)


def draw_audio_local(rng: np.random.Generator, T: int, T_donor: int,
                     law: PlacementLaw,
                     shift_rng: np.random.Generator,
                     ) -> tuple[str, dict, Segments]:
    """Draw one audio-local corruption (kind, parameters, inclusive
    segment bounds) without touching any feature, so the same drawn
    corruption applies to both feature spaces of a clip."""
    u = rng.random()
    if u < AUDIO_REPLACE_FRACTION:
        reps, segs = [], []
        for _ in range(int(rng.integers(1, 3))):
            s, L = segment(rng, T, law)
            o = int(rng.integers(0, max(1, T_donor - L)))
            reps.append((s, L, o))
            segs.append((s, s + L - 1))
        return "a_replace", {"replacements": reps}, segs
    s, L = segment(rng, T, law)
    rng.choice(_RETIRED_AUDIO_SHIFT_FRAMES)     # drawn and discarded
    d = (int(shift_rng.integers(1, T))
         * (1 if shift_rng.random() < 0.5 else -1))
    return "a_shift", {"segment": (s, L), "shift": d}, [(s, s + L - 1)]


def apply_audio_local(audio: np.ndarray, joint: np.ndarray,
                      donor_audio: np.ndarray | None,
                      donor_joint: np.ndarray | None, kind: str,
                      params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply an audio replacement or edge-clamped shift; leave visual features unchanged."""

    A, J = audio.copy(), joint.copy()
    T = len(A)
    if kind == "a_replace":
        if donor_audio is None or donor_joint is None:
            raise ValueError(
                "a_replace needs a donor in both passes; got "
                f"donor_audio={donor_audio is not None}, "
                f"donor_joint={donor_joint is not None}")
        for s, L, o in params["replacements"]:
            n_donor = min(len(donor_audio), len(donor_joint))
            if o + L > n_donor:
                raise ValueError(
                    f"a_replace donor slice [{o}, {o + L}) exceeds donor "
                    f"length {n_donor} for target [{s}, {s + L})")
            A[s:s + L] = donor_audio[o:o + L]
            J[s:s + L] = 0.5 * J[s:s + L] + 0.5 * donor_joint[o:o + L]
    elif kind == "a_shift":
        s, L = params["segment"]
        idx = np.clip(np.arange(s, s + L) + params["shift"], 0, T - 1)
        A[s:s + L] = A[idx]
    else:
        raise ValueError(f"unknown audio-local kind '{kind}'")
    return A, J
