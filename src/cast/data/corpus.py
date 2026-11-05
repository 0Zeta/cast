#!/usr/bin/env python3
"""Build a self-splice corpus from genuine-source extractions."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-clean", required=True)
    parser.add_argument("--auxiliary-clean", required=True)
    parser.add_argument("--primary-visual-blend", required=True)
    parser.add_argument("--auxiliary-visual-blend", required=True)
    parser.add_argument("--primary-audio-blend", required=True)
    parser.add_argument("--auxiliary-audio-blend", required=True)
    parser.add_argument("--source-manifest", required=True,
                        help="CSV with video_id and explicit speaker_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-test matched-pair limit; zero means all")

    root = Path(__file__).resolve().parents[3]
    checkpoints = root / "models" / "checkpoints"
    parser.add_argument("--prediction-heads",
                        default=str(checkpoints / "prediction_heads.pt"))
    parser.add_argument("--visual-crossfit", nargs=2, metavar=("FOLD0", "FOLD1"),
                        default=[str(checkpoints / "visual_artifact_crossfit_0.pt"),
                                 str(checkpoints / "visual_artifact_crossfit_1.pt")])
    parser.add_argument("--audio-crossfit", nargs=2, metavar=("FOLD0", "FOLD1"),
                        default=[str(checkpoints / "audio_artifact_crossfit_0.pt"),
                                 str(checkpoints / "audio_artifact_crossfit_1.pt")])
    args = parser.parse_args(argv)

    from cast.extract.corpus import build_corpus_from_features

    build_corpus_from_features(
        args.primary_clean,
        args.auxiliary_clean,
        args.primary_visual_blend,
        args.auxiliary_visual_blend,
        args.primary_audio_blend,
        args.auxiliary_audio_blend,
        args.source_manifest,
        args.prediction_heads,
        tuple(args.visual_crossfit),
        tuple(args.audio_crossfit),
        args.output,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
