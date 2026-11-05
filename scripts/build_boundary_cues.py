#!/usr/bin/env python3
"""Build proposal audio boundary cues from raw videos."""

import argparse

from cast.ranking.build import build_boundary_cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--clip-scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curves-dir")
    parser.add_argument("--curve-video-ids", nargs="+")
    args = parser.parse_args()
    build_boundary_cues(args.proposals, args.clip_scores, args.labels, args.manifest,
                        args.output, curves_dir=args.curves_dir,
                        curve_video_ids=args.curve_video_ids)


if __name__ == "__main__":
    main()
