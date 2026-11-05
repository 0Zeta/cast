#!/usr/bin/env python3
"""Scan and evaluate a CAST protocol, or evaluate an existing run."""

import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", help="protocol name or JSON path")
    parser.add_argument("--run", help="existing run; skips fitting and scanning")
    parser.add_argument("--out", required=True)
    parser.add_argument("--store")
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--bootstrap", nargs="?", type=int, const=20000, default=0)
    parser.add_argument("--figure-cases")
    parser.add_argument("--manifest", help="media CSV used for audio boundary cues and figures")
    args = parser.parse_args()
    if not args.protocol and not args.run:
        parser.error("provide --protocol or --run")
    from cast.workflow import evaluate
    output = evaluate(args.protocol, args.out, run=args.run, store=args.store,
                      nproc=args.nproc, bootstrap=args.bootstrap,
                      figure_cases=args.figure_cases, manifest=args.manifest)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
