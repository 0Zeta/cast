#!/usr/bin/env python3
"""Fit the cross-clip proposal ranker on self-splice proposals."""

import argparse

from cast.ranking.build import fit_proposal_ranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--boundary-cues", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    fit_proposal_ranker(args.proposals, args.corpus, args.boundary_cues, args.output,
                        features=args.features, protocol=args.protocol)


if __name__ == "__main__":
    main()
