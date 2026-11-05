#!/usr/bin/env python3
"""Rebuild the paper's frame strip while preserving its measured waveform."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


FRAME_IDS = tuple(range(30, 54, 2))


def main() -> None:
    root = Path(__file__).resolve().parent
    frame_dir = root / "assets" / "frames"
    frames = [
        frame_dir / f"frame_{index:03d}.png"
        for index in FRAME_IDS
    ]
    waveform = root / "assets" / "audio_waveform.tex"
    missing = [str(path) for path in [*frames, waveform] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing paper illustration assets: {missing}")

    with Image.open(frames[0]) as frame:
        width, height = frame.size
    strip = Image.new("RGB", (width, height * len(frames)))
    for index, path in enumerate(frames):
        with Image.open(path) as frame:
            if frame.size != (width, height):
                raise ValueError(f"inconsistent paper frame dimensions: {path}")
            strip.paste(frame, (0, index * height))
    strip.save(root / "assets" / "frames_strip_vertical.png")


if __name__ == "__main__":
    main()
