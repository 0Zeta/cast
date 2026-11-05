#!/usr/bin/env python3
"""Extract encoder features or construct frame-level score caches."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("features", "scores"))
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return
    args = parser.parse_args(argv[:1])
    from cast.extract.cli import build_scores, extract_features
    (extract_features if args.stage == "features" else build_scores)(argv[1:])


if __name__ == "__main__":
    main()
