"""Reconstruct the audio of self-splice rows for audio boundary cue extraction."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

SAMPLES_PER_FRAME = 640          # 16 kHz audio against 25-fps video
SHIFT_FRAMES = 5                 # circular shift of own audio, in frames

AUDIO_SPLICE_KINDS = ("ablend", "a_replace", "a_shift")
DONOR_KINDS = ("ablend", "a_replace")


def media_paths(manifest: str | Path) -> dict[str, Path]:
    """video_id -> media path from a manifest CSV; relative paths resolve
    against the CSV's directory after environment-variable expansion."""
    path = Path(manifest)
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("video_id", "")).strip()
            source = str(row.get("video_path", "")).strip()
            if not video_id or not source:
                raise ValueError(f"{path}: a row lacks video_id or video_path")
            if video_id in result:
                raise ValueError(f"{path}: duplicate video_id {video_id!r}")
            source_path = Path(os.path.expandvars(source))
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            result[video_id] = source_path.resolve()
    return result


def decode_audio(path: str | Path) -> np.ndarray:
    """Mono 16-kHz float32 waveform in [-1, 1], decoded with ffmpeg."""
    process = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, timeout=180, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: "
                           f"{process.stderr.decode(errors='replace')[:200]}")
    waveform = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32)
    if not len(waveform):
        raise RuntimeError(f"no audio samples decoded from {path}")
    return waveform / 32768.0


def donor_sources(kinds, sources) -> list[str | None]:
    """The donor source clip of each row: the source of the preceding
    donor-kind row in corpus order, or None before the first."""
    donors: list[str | None] = [None] * len(kinds)
    previous = None
    for i, (kind, source) in enumerate(zip(kinds, sources)):
        if kind in DONOR_KINDS:
            donors[i] = previous
            previous = source or previous
    return donors


def spliced_waveform(source: np.ndarray, segments, kind: str,
                     donor: np.ndarray | None) -> np.ndarray:
    """The waveform stand-in of one corpus row.

    Audio-splice rows replace each synthetic segment with the donor's
    centered slice at native gain and a hard cut; ``a_shift`` rows, and
    donor-kind rows without a long enough donor, use their own audio
    shifted circularly by SHIFT_FRAMES. Other rows keep the source audio."""
    if kind not in AUDIO_SPLICE_KINDS or not len(segments):
        return source
    out = source.copy()
    n = len(out)
    for a, b in segments:
        s0 = min(SAMPLES_PER_FRAME * int(a), n)
        s1 = min(SAMPLES_PER_FRAME * (int(b) + 1), n)
        length = s1 - s0
        if length <= 0:
            continue
        if kind == "a_shift" or donor is None or len(donor) < length:
            idx = (np.arange(s0, s1) + SHIFT_FRAMES * SAMPLES_PER_FRAME) % n
            out[s0:s1] = source[idx]
        else:
            offset = max(0, (len(donor) - length) // 2)
            out[s0:s1] = donor[offset:offset + length]
    return out
