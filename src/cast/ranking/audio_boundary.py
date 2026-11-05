"""Extract audio boundary cues from full-band positive log-power flux."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
STFT_WINDOW = 512
STFT_HOP = 160                 # 10 ms -> 100 Hz curve rate
STEPS_PER_VIDEO_FRAME = 4      # 25 fps video against a 100 Hz curve
EDGE_STEPS = 8                 # +/-80 ms edge window
MIN_STEPS = 12                 # below this a clip yields no curves
_EPS = 1e-12


def column_names() -> tuple[str, ...]:
    """The adopted proposal cue, e_I."""
    return ("audio_boundary_cue",)


def power_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """(n_steps, n_bins) power spectrogram; raises if the clip is short."""
    n = 1 + max(0, len(waveform) - STFT_WINDOW) // STFT_HOP
    if n < MIN_STEPS:
        raise ValueError(
            f"clip yields {n} STFT steps, fewer than the {MIN_STEPS} the "
            f"curves need ({len(waveform)} samples at {SAMPLE_RATE} Hz)")
    idx = (np.arange(STFT_WINDOW)[None, :]
           + STFT_HOP * np.arange(n)[:, None])
    frames = waveform[idx] * np.hanning(STFT_WINDOW)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2


def percentile_rank(curve: np.ndarray) -> np.ndarray:
    """Plain within-clip ECDF, ``(rank + 1) / n``, with stable ties."""
    order = np.argsort(np.argsort(curve, kind="stable"), kind="stable")
    return (order + 1) / len(curve)


def curves(waveform: np.ndarray) -> np.ndarray:
    """(n_steps, 1) full-band flux percentile curve.

    The proposal audio boundary cue averages its available boundary-window
    maxima in proposal_features.
    """
    power = power_spectrogram(waveform)
    n = len(power)
    log_power = np.log(power + _EPS)
    flux = np.diff(log_power, axis=0)
    flux[flux < 0] = 0.0                      # rectify

    full = np.zeros(n)
    full[1:] = flux.sum(1)
    return percentile_rank(full).astype(np.float32)[:, None]


def proposal_features(ranked: np.ndarray, start_frame: np.ndarray,
                      end_frame: np.ndarray, *,
                      require_edge: bool = True) -> np.ndarray:
    """Return the audio boundary cue as one column per proposal.

    Average the maxima of the available edge windows. A proposal with
    neither edge window inside the curve raises, or yields NaN when
    ``require_edge`` is off.
    """
    n_steps = len(ranked)
    out = np.full((len(start_frame), 1), np.nan, np.float32)

    def window_max(lo: int, hi: int) -> float | None:
        lo, hi = max(0, lo), min(n_steps, hi)
        return None if lo >= hi else float(ranked[lo:hi, 0].max())

    for k, (a, b) in enumerate(zip(start_frame, end_frame)):
        ca = STEPS_PER_VIDEO_FRAME * int(a)
        cb = STEPS_PER_VIDEO_FRAME * (int(b) + 1)
        ea = window_max(ca - EDGE_STEPS, ca + EDGE_STEPS + 1)
        eb = window_max(cb - EDGE_STEPS, cb + EDGE_STEPS + 1)
        edges = [v for v in (ea, eb) if v is not None]
        if not edges:
            if require_edge:
                raise ValueError(
                    f"proposal {k} (frames {int(a)}..{int(b)}) has "
                    f"neither edge window inside a curve of "
                    f"{n_steps} steps")
            continue
        out[k, 0] = sum(edges) / len(edges)
    return out


@dataclass(frozen=True)
class ClipBoundaryCues:
    """One clip's packed rows, ready to concatenate into a store."""

    start_frame: np.ndarray
    end_frame: np.ndarray
    features: np.ndarray

    @property
    def column_names(self) -> tuple[str, ...]:
        return column_names()


def extract_clip(waveform: np.ndarray, start_frame: np.ndarray,
                 end_frame: np.ndarray, *,
                 require_edge: bool = True) -> ClipBoundaryCues:
    """Full path for one clip: waveform + proposal geometry -> rows."""
    start_frame = np.asarray(start_frame, dtype=np.int64)
    end_frame = np.asarray(end_frame, dtype=np.int64)
    if len(start_frame) != len(end_frame):
        raise ValueError("start_frame and end_frame differ in length")
    if np.any(end_frame < start_frame):
        raise ValueError("a proposal ends before it starts")
    feats = proposal_features(curves(waveform), start_frame, end_frame,
                              require_edge=require_edge)
    return ClipBoundaryCues(start_frame=start_frame, end_frame=end_frame,
                       features=feats)
