#!/usr/bin/env python3
"""Run one published CAST ablation with its scan and ranker fit policy."""

import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", help="baseline emission for ranking-only arms")
    parser.add_argument("--store")
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--bootstrap", nargs="?", type=int, const=20000, default=0)
    parser.add_argument("--figure-cases")
    parser.add_argument("--manifest")
    parser.add_argument("--self-splice-manifest")
    args = parser.parse_args()
    from cast.workflow import run_ablation
    output = run_ablation(args.protocol, args.arm, args.out, run=args.run,
                          store=args.store, nproc=args.nproc, bootstrap=args.bootstrap,
                          figure_cases=args.figure_cases, manifest=args.manifest,
                          self_splice_manifest=args.self_splice_manifest)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
