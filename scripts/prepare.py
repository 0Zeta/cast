#!/usr/bin/env python3
"""Prepare dataset manifests, external assets, and controlled corruptions."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "evaluation": ("avdeepfake", "AV-Deepfake1M or AV-Deepfake1M++ validation manifest"),
    "reference": ("reference", "ordered AV-Deepfake1M genuine reference manifest"),
    "sources": ("sources", "controlled-corruption source and exclusion manifests"),
    "transfer": ("transfer", "LAV-DF, FakeAVCeleb, or AVLips manifests"),
    "assets": ("assets", "pretrained encoders and mouth-alignment assets"),
    "corpus": ("corpus", "controlled-corruption corpus from six feature extractions"),
    "durations": ("durations", "duration models from aligned genuine speech"),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog="\n".join(
        f"  {name}: {description}" for name, (_, description) in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return
    args = parser.parse_args(argv[:1])
    module = importlib.import_module(f"cast.data.{COMMANDS[args.command][0]}")
    module.main(argv[1:])


if __name__ == "__main__":
    main()
