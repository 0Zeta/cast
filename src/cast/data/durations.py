#!/usr/bin/env python3
"""Fit and store genuine-speech duration models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from cast.fits.records import (duration_law_record,     # noqa: E402
                               max_duration_record,
                               write_fit_record)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fit the genuine-only duration objects and write their "
            "deterministic records."
        ),
    )
    ap.add_argument(
        "--word-alignments",
        required=True,
        metavar="PATH",
        help="path to the genuine-word-alignment CSV",
    )
    ap.add_argument(
        "--out-dir",
        default="models",
        metavar="DIR",
        help="directory for fitted-object records (default: models)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    for name, rec in (
            ("duration_law", duration_law_record(args.word_alignments)),
            ("max_duration_frames",
             max_duration_record(args.word_alignments))):
        write_fit_record(out / f"{name}.json", rec)
        print(f"wrote {out / f'{name}.json'}")


if __name__ == "__main__":
    main()
