#!/usr/bin/env python3
"""Train neural heads or fit and scan a CAST protocol."""

import argparse
import sys

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"heads", "feature-cache"}:
        from cast.training.cli import cache_features, train_heads
        (train_heads if argv[0] == "heads" else cache_features)(argv[1:])
        return
    if argv and argv[0] == "protocol":
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Stages: heads trains neural heads; feature-cache builds a memory-mapped "
        "feature cache; protocol (the default) fits reference models and scans. "
        "Use 'heads --help' or 'feature-cache --help' for their arguments."))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--store")
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--fit-ranker", action="store_true")
    parser.add_argument("--self-splice-manifest", help="genuine source media CSV for ranker fitting")
    parser.add_argument("--figure-cases")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    from cast.workflow import train
    output = train(args.protocol, args.out, store=args.store, nproc=args.nproc,
                   fit_ranker=args.fit_ranker, self_splice_manifest=args.self_splice_manifest,
                   figure_cases=args.figure_cases, manifest=args.manifest)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
