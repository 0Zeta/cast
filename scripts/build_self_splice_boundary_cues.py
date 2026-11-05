#!/usr/bin/env python3
"""Build self-splice proposal audio boundary cues from source audio."""

import argparse

from cast.ranking.build import build_self_splice_boundary_cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-record")
    args = parser.parse_args()
    build_self_splice_boundary_cues(args.proposals, args.corpus, args.manifest,
                                    args.output, build_record=args.build_record)


if __name__ == "__main__":
    main()
