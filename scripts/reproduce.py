#!/usr/bin/env python3
"""Run the CAST pipeline for a published protocol."""

from __future__ import annotations

import argparse


def main(argv=None):
    from cast.reproduce import STAGES, reproduce

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--store")
    parser.add_argument("--from", dest="start_at", choices=STAGES, default="prepare")
    parser.add_argument("--extract-python", help="Python 3.10 executable for AV-HuBERT")
    parser.add_argument("--avhubert-root")
    parser.add_argument("--assets")
    parser.add_argument("--head-dir")
    parser.add_argument("--train-heads", action="store_true")
    parser.add_argument("--head-config")
    parser.add_argument("--fit-ranker", action="store_true")
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", nargs="?", type=int, const=20000, default=0)
    parser.add_argument("--figure-cases")
    parser.add_argument("--figures", help="figure target, such as all or calibration")
    parser.add_argument("--formats", default="pdf")
    args = parser.parse_args(argv)
    print(f"wrote {reproduce(**vars(args))}")


if __name__ == "__main__":
    main()
