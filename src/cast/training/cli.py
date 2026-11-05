"""Commands for neural heads and memory-mapped training features."""

from __future__ import annotations

import argparse
import json

from .config import load_head_config, validate_head_config


def cache_features(argv=None):
    parser = argparse.ArgumentParser(description="Build a memory-mapped training feature cache.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True, help="output prefix")
    parser.add_argument("--with-joint", action="store_true")
    args = parser.parse_args(argv)
    from .data import build_feature_cache
    print(build_feature_cache(args.features, args.out, args.with_joint))


def train_heads(argv=None):
    parser = argparse.ArgumentParser(description="Train the alignment stack and modality-specific artifact heads on genuine frozen features.",
                                     allow_abbrev=False)
    parser.add_argument("--config", help="scientific settings JSON (defaults to configs/heads.json)")
    parser.add_argument("--features", required=True, help="clean training feature TAR directory or cache prefix")
    parser.add_argument("--visual-blend", required=True, help="pixel-space visual-blend feature TAR directory")
    parser.add_argument("--audio-blend", required=True, help="filterbank audio-blend feature TAR directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--steps-align", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--steps-ref", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--steps-pred", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--window-frames", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--validation-batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--pca-train-batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--pca-validation-batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--prediction-validation-batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        settings = load_head_config(args.config)
        settings.update({name: getattr(args, name) for name in settings
                         if hasattr(args, name)})
        validate_head_config(settings)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    vars(args).update(settings)

    from pathlib import Path
    from .data import FeatureCache
    from .prediction import train_prediction_heads
    from .artifacts import train_artifact_heads

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training.json").write_text(json.dumps(settings, indent=2) + "\n")
    features = FeatureCache(args.features)
    train_prediction_heads(
        features, output / "prediction_heads.pt", device=args.device,
        steps_align=args.steps_align, steps_ref=args.steps_ref, steps_pred=args.steps_pred,
        batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
        window_frames=args.window_frames, preload=args.preload, augment=args.augment,
        validation_batches=args.validation_batches, pca_train_batches=args.pca_train_batches,
        pca_validation_batches=args.pca_validation_batches,
        prediction_validation_batches=args.prediction_validation_batches)
    options = {"epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed}
    train_artifact_heads(features, args.visual_blend, args.audio_blend, output,
                         visual=options, audio=options, device=args.device)
    print(f"wrote heads to {output}")
