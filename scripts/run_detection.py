#!/usr/bin/env python3
"""Run CAST from score caches and JSON configuration."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cast.config import (CastConfig,              # noqa: E402
                         load_config_document)
from cast.scan import ScanInputs, run_detection  # noqa: E402


def load_config_argument(path: str | Path) -> CastConfig:
    """Load the interval-scan configuration."""
    return load_config_document(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run CAST detection from local score caches."
        ),
    )
    ap.add_argument(
        "--inputs",
        required=True,
        metavar="PATH",
        help="path to the ScanInputs JSON document",
    )
    ap.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="method configuration JSON",
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=None,
        metavar="N",
        help="override the configured worker-process count",
    )
    ap.add_argument(
        "--stage-dump-dir",
        default=None,
        metavar="DIR",
        help="optionally write intermediate stage outputs to this directory",
    )
    args = ap.parse_args(argv)

    inputs = ScanInputs.load(args.inputs)
    config = load_config_argument(args.config)
    if args.nproc is not None:
        config = dataclasses.replace(
            config, execution=dataclasses.replace(config.execution,
                                                  nproc=args.nproc))
    t0 = time.time()
    out_dir = run_detection(inputs, config,
                            stage_dump_dir=args.stage_dump_dir)
    print(f"total wall: {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
