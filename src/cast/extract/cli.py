"""Feature extraction and frame-level score cache commands."""

from __future__ import annotations

import argparse
from pathlib import Path

def extract_features(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract frozen AV-HuBERT representations.")
    parser.add_argument("--manifest", required=True,
                        help="CSV with video_id and video_path columns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--avhubert-root", required=True,
                        help="facebookresearch/av_hubert checkout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cnn-detector", required=True,
                        help="mmod_human_face_detector.dat")
    parser.add_argument("--face-predictor", required=True,
                        help="shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--mean-face", required=True,
                        help="20words_mean_face.npy")
    parser.add_argument("--mode", default="clean",
                        choices=("clean", "visual-blend", "audio-blend"))
    parser.add_argument("--blend-purpose", default="corpus",
                        choices=("corpus", "artifact-head"),
                        help="select the corpus or artifact-training self-blend duration law")
    parser.add_argument(
        "--leading-silence-trim", action="store_true",
        help="AVH-Align/AVPF evaluation protocol: remove audio onset "
             "silence from both modalities in the selected trim order")
    parser.add_argument("--trim-order", default="trim_then_cap",
                        choices=("cap_then_trim", "trim_then_cap"),
                        help="order onset trimming and the 750-frame cap; "
                             "shared10k uses cap_then_trim")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--chunk-clips", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from cast.extract.avhubert import extract_manifest

    result = extract_manifest(
        args.manifest,
        args.output_dir,
        args.avhubert_root,
        args.checkpoint,
        args.cnn_detector,
        args.face_predictor,
        args.mean_face,
        mode=args.mode,
        rank=args.rank,
        nshard=args.nshard,
        chunk_clips=args.chunk_clips,
        log_every=args.log_every,
        device=args.device,
        leading_silence_trim=args.leading_silence_trim,
        trim_order=args.trim_order,
        blend_purpose=args.blend_purpose,
    )
    print(result)


def build_scores(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply the heads and construct frame-level score caches.")
    parser.add_argument("--primary-features", required=True,
                        help="large_vox_iter5.pt feature-tar directory")
    parser.add_argument("--auxiliary-features", required=True,
                        help="self_large_vox_433h.pt feature-tar directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="evaluation")
    parser.add_argument("--device", default="cuda")
    root = Path(__file__).resolve().parents[3]
    checkpoints = root / "models" / "checkpoints"
    parser.add_argument("--prediction-heads",
                        default=str(checkpoints / "prediction_heads.pt"))
    parser.add_argument("--visual-artifact-head",
                        default=str(checkpoints / "visual_artifact_head.pt"))
    parser.add_argument("--audio-artifact-head",
                        default=str(checkpoints / "audio_artifact_head.pt"))
    args = parser.parse_args(argv)

    from cast.extract import build_runtime_caches

    build_runtime_caches(
        args.primary_features,
        args.auxiliary_features,
        args.prediction_heads,
        args.visual_artifact_head,
        args.audio_artifact_head,
        args.output_dir,
        prefix=args.prefix,
        device=args.device,
    )
