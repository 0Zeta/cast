#!/usr/bin/env python3
"""Rank emitted proposals with the cross-clip proposal ranker and audio boundary cues."""

import argparse

from cast.config import RankingConfig
from cast.ranking.build import _rank_proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--clip-scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--boundary-cues", help="cue store; required by audio features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-proposal-width", type=int,
                        default=RankingConfig().min_proposal_width)
    args = parser.parse_args()
    _rank_proposals(args.proposals, args.clip_scores, args.labels, args.ranker,
                    args.boundary_cues, args.output,
                    RankingConfig(min_proposal_width=args.min_proposal_width))


if __name__ == "__main__":
    main()
