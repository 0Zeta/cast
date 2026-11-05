"""Run the CAST reproduction stages for a declared protocol."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

from . import workflow
from .fits.heads import AUXILIARY_HEADS, CORPUS_CROSSFIT_HEADS

REPO = Path(__file__).resolve().parents[2]
STAGES = ("prepare", "features", "heads", "scores", "corpus", "fit", "evaluate", "figures")


def _prepare(command, arguments):
    module = {"evaluation": "avdeepfake", "reference": "reference",
              "sources": "sources", "transfer": "transfer"}[command]
    importlib.import_module(f"cast.data.{module}").main(list(map(str, arguments)))


def _environment_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"set {name} before preparing this dataset")
    return Path(value).expanduser().resolve()


def _av_manifest(dataset, output, *, cohort=None, complete=False):
    variable = "AVDEEPFAKE1MPP_ROOT" if dataset == "avdeepfake1m-plus" else "AVDEEPFAKE1M_ROOT"
    root = _environment_path(variable)
    args = ["--dataset", dataset, "--metadata", root / "val_metadata.json",
            "--dataset-root", root, "--output", output]
    if cohort:
        args += ["--cohort", cohort]
    if complete:
        args += ["--include-silent-videos"]
    _prepare("evaluation", args)


def _prefix(path):
    name = Path(path).name
    if not name.endswith("_main.npz"):
        raise ValueError(f"main cache name must end with _main.npz: {path}")
    return name[:-len("_main.npz")]


def prepare_inputs(protocol, store, reference_manifest, source_manifest, exclusion):
    """Prepare the fixed dataset populations through their existing commands."""
    inputs = protocol["inputs"]
    labels = Path(inputs["labels_csv"])
    name = protocol["name"]
    if not labels.is_file():
        if name.startswith("avdeepfake1m_plus"):
            _av_manifest("avdeepfake1m-plus", labels)
        elif name.startswith("avdeepfake1m"):
            cohort = "avh-align-avpf-10000" if "shared_10000" in name else None
            _av_manifest("avdeepfake1m", labels, cohort=cohort)
        elif name == "lav_df":
            root = _environment_path("LAVDF_ROOT")
            _prepare("transfer", ["lavdf", "--root", root,
                                 "--metadata", root / "metadata.min.json", "--output", labels])
        elif name == "fakeavceleb_voxceleb2":
            _prepare("transfer", ["fakeavceleb-voxceleb2",
                                 "--fakeavceleb-root", _environment_path("FAKEAVCELEB_ROOT"),
                                 "--voxceleb2-root", _environment_path("VOXCELEB2_ROOT"),
                                 "--output", labels])
        elif name == "fakeavceleb_avpf_trimmed":
            _prepare("transfer", ["fakeavceleb-avpf",
                                 "--fakeavceleb-root", _environment_path("FAKEAVCELEB_ROOT"),
                                 "--output", labels])
        elif name in ("avlips", "avlips_trimmed"):
            _prepare("transfer", ["avlips", "--root", _environment_path("AVLIPS_ROOT"),
                                 "--output", labels])
        else:
            raise ValueError(f"prepare a labels manifest for custom protocol {name!r}")

    complete = protocol["evaluation"].get("localization")
    if isinstance(complete, dict) and complete.get("complete_labels"):
        path = Path(complete["complete_labels"])
        if not path.is_file():
            _av_manifest("avdeepfake1m-plus", path, complete=True)

    if not Path(reference_manifest).is_file():
        validation = store / "manifests/avdeepfake1m_validation.csv"
        validation_plus = store / "manifests/avdeepfake1m_plus_validation.csv"
        for dataset, output in (("avdeepfake1m", validation),
                                ("avdeepfake1m-plus", validation_plus)):
            if not output.is_file():
                _av_manifest(dataset, output)
        _prepare("reference", [
            "--train-metadata", _environment_path("AVDEEPFAKE1M_ROOT") / "train_metadata.json",
            "--exclude-validation-manifest", validation,
            "--exclude-validation-manifest", validation_plus,
            "--output", reference_manifest])
    if not Path(source_manifest).is_file() or not Path(exclusion).is_file():
        _prepare("sources", ["--genuine-manifest", reference_manifest,
                             "--train-metadata", _environment_path("AVDEEPFAKE1M_ROOT") / "train_metadata.json",
                             "--sources-out", source_manifest,
                             "--reference-exclusion-out", exclusion])


def _extract(manifest, output, checkpoint, assets, avhubert_root, python,
             *, device, mode="clean", purpose="corpus", nshard=1,
             leading_silence_trim=False, trim_order="trim_then_cap"):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPO / "src"), environment.get("PYTHONPATH"))))
    for rank in range(nshard):
        command = [
            str(python), str(REPO / "scripts/extract.py"), "features",
            "--manifest", str(manifest), "--output-dir", str(output),
            "--avhubert-root", str(avhubert_root), "--checkpoint", str(checkpoint),
            "--cnn-detector", str(assets / "mmod_human_face_detector.dat"),
            "--face-predictor", str(assets / "shape_predictor_68_face_landmarks.dat"),
            "--mean-face", str(assets / "20words_mean_face.npy"),
            "--mode", mode, "--blend-purpose", purpose,
            "--rank", str(rank), "--nshard", str(nshard), "--device", device]
        if leading_silence_trim:
            command += ["--leading-silence-trim", "--trim-order", trim_order]
        subprocess.run(command, check=True, env=environment)


def _build_scores(protocol, store, heads, *, device):
    from .extract.scores import build_runtime_caches
    inputs = protocol["inputs"]
    for reference in (False, True):
        key = "ref_scores_main" if reference else "scores_main"
        if not inputs.get(key):
            continue
        prefix = _prefix(inputs[key])
        output = Path(inputs[key]).parent
        for group in ("main", "avsr", "sync", "artifact"):
            field = ("ref_" if reference else "") + f"scores_{group}"
            if Path(inputs[field]) != output / f"{prefix}_{group}.npz":
                raise ValueError(f"score-cache paths must share a directory and prefix: {field}")
        build_runtime_caches(
            store / f"features/{prefix}-primary",
            store / f"features/{prefix}-auxiliary",
            heads / "prediction_heads.pt", heads / "visual_artifact_head.pt",
            heads / "audio_artifact_head.pt", output, prefix=prefix, device=device)


def reproduce(protocol, out, *, store=None, start_at="prepare",
              extract_python=None, avhubert_root=None, assets=None,
              head_dir=None, train_heads=False, head_config=None,
              fit_ranker=False, nproc=None, device="cuda", bootstrap=0,
              figure_cases=None, figures=None, formats="pdf",
              corpus_limit=0, log=print) -> Path:
    """Prepare, extract, fit and evaluate one protocol with its fixed settings."""
    if start_at not in STAGES:
        raise ValueError(f"start_at must be one of {STAGES}")
    protocol = workflow.load_protocol(protocol, store=store)
    inputs = protocol["inputs"]
    store = Path(inputs["base_dir"])
    out = Path(out).expanduser().resolve()
    assets = Path(assets).expanduser().resolve() if assets else store / "assets"
    heads = (Path(head_dir).expanduser().resolve() if head_dir else
             store / "heads" if train_heads else REPO / "models/checkpoints")
    source_manifest = Path(protocol["ranking"].get("self_splice_manifest")
                           or store / "manifests/self_splice_sources.csv")
    reference_manifest = Path(inputs.get("ref_labels_csv")
                              or store / "manifests/avdeepfake1m_reference.csv")
    exclusion = Path(inputs.get("ref_exclude_csv")
                     or store / "manifests/reference_exclusion.csv")
    reference_prefix = (_prefix(inputs["ref_scores_main"])
                        if inputs.get("ref_scores_main") else "avdeepfake1m_reference")
    media_manifest = Path(protocol["evaluation"].get("manifest") or inputs["labels_csv"])
    selected = set(STAGES[STAGES.index(start_at):])
    if figure_cases and protocol["evaluation"].get("localization", True) is False:
        raise ValueError("figure cases require a localization-enabled protocol")
    if "features" in selected and (not extract_python or not avhubert_root):
        raise ValueError("feature extraction needs --extract-python and --avhubert-root")
    if corpus_limit < 0:
        raise ValueError("corpus_limit must be nonnegative")
    if head_config and not train_heads:
        raise ValueError("--head-config requires --train-heads")
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if "fit" in selected and (out / "clip_scores.csv").exists():
        raise FileExistsError(f"scan already exists: {out}; use --from evaluate")
    if train_heads and "heads" in selected:
        roles = (*AUXILIARY_HEADS, *CORPUS_CROSSFIT_HEADS)
        if any((heads / f"{role}.pt").exists() for role in roles):
            raise FileExistsError(f"head checkpoints already exist: {heads}; use --from scores")
        from .training.config import load_head_config
        load_head_config(head_config)

    if "prepare" in selected:
        log("Prepare manifests and assets")
        prepare_inputs(protocol, store, reference_manifest, source_manifest, exclusion)
        from .data.assets import download_assets
        download_assets(assets, REPO / "configs/assets.json")

    if "features" in selected:
        log("Extract encoder features")
        preprocessing = protocol.get("preprocessing", {})
        encoders = (("primary", "large_vox_iter5.pt"),
                    ("auxiliary", "self_large_vox_433h.pt"))
        pools = [(media_manifest, _prefix(inputs["scores_main"]), True, encoders)]
        if inputs.get("ref_scores_main") or train_heads:
            reference_encoders = encoders if inputs.get("ref_scores_main") else encoders[:1]
            pools.append((reference_manifest, reference_prefix, False, reference_encoders))
        for manifest, prefix, evaluation, selected_encoders in pools:
            for encoder, checkpoint in selected_encoders:
                _extract(
                    manifest, store / f"features/{prefix}-{encoder}",
                    assets / checkpoint, assets, avhubert_root, extract_python,
                    device=device,
                    leading_silence_trim=evaluation and preprocessing.get("leading_silence_trim", False),
                    trim_order=preprocessing.get("trim_order", "trim_then_cap"))
        for encoder, checkpoint in (("primary", "large_vox_iter5.pt"),
                                    ("auxiliary", "self_large_vox_433h.pt")):
            for suffix, mode in (("clean", "clean"), ("visual", "visual-blend"),
                                 ("audio", "audio-blend")):
                _extract(
                    source_manifest, store / f"features/self-splice-{encoder}-{suffix}",
                    assets / checkpoint, assets, avhubert_root, extract_python,
                    device=device, mode=mode, nshard=9)
        if train_heads:
            for stream in ("visual", "audio"):
                _extract(
                    source_manifest, store / f"features/artifact-{stream}",
                    assets / "large_vox_iter5.pt", assets, avhubert_root, extract_python,
                    device=device, mode=f"{stream}-blend", purpose="artifact-head", nshard=9)

    if "heads" in selected and train_heads:
        log("Train neural heads")
        from .training.cli import train_heads as train_command
        arguments = [
            "--features", str(store / f"features/{reference_prefix}-primary"),
            "--visual-blend", str(store / "features/artifact-visual"),
            "--audio-blend", str(store / "features/artifact-audio"),
            "--out", str(heads), "--device", device]
        if head_config:
            arguments += ["--config", str(head_config)]
        train_command(arguments)

    if "scores" in selected:
        log("Build runtime score caches")
        _build_scores(protocol, store, heads, device=device)

    if "corpus" in selected:
        log("Build the controlled-corruption corpus")
        from .extract.corpus import build_corpus_from_features
        build_corpus_from_features(
            store / "features/self-splice-primary-clean",
            store / "features/self-splice-auxiliary-clean",
            store / "features/self-splice-primary-visual",
            store / "features/self-splice-auxiliary-visual",
            store / "features/self-splice-primary-audio",
            store / "features/self-splice-auxiliary-audio",
            source_manifest, heads / "prediction_heads.pt",
            tuple(heads / f"visual_artifact_crossfit_{fold}.pt" for fold in range(2)),
            tuple(heads / f"audio_artifact_crossfit_{fold}.pt" for fold in range(2)),
            inputs["self_splice_corpus"], device=device, limit=corpus_limit, log=log)

    if "fit" in selected:
        log("Fit reference models and scan")
        workflow.train(
            protocol, out, nproc=nproc, fit_ranker=fit_ranker or train_heads,
            self_splice_manifest=source_manifest, figure_cases=figure_cases,
            manifest=media_manifest, log=log)
    if "evaluate" in selected:
        log("Evaluate the saved run")
        workflow.evaluate(
            run=out, out=out, bootstrap=bootstrap,
            figure_cases=figure_cases, manifest=media_manifest, log=log)
    target = figures or ("all" if figure_cases or start_at == "figures" else None)
    if "figures" in selected and target:
        log("Render figures")
        subprocess.run([
            sys.executable, str(REPO / "figures/build.py"), "build", target,
            "--run", str(out), "--formats", formats], check=True)
    return out
