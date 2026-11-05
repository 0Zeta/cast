"""Fit, scan, rank, and evaluate explicit CAST protocols."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .config import CastConfig
from .eval import runner
from .scan import ScanInputs, run_detection

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO / "configs/protocols"
ABLATIONS = REPO / "configs/ablations/definitions.json"
FEATURES = ("log1p_rank", "adjusted_score", "audio_boundary_cue")
PROTOCOL_DETAILS = ("references", "population", "preprocessing", "fitting")


def _json(path):
    return json.loads(Path(path).read_text())


def _path(value, base):
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "${" in expanded:
        raise ValueError(f"unresolved data root in {value!r}; pass --store")
    path = Path(expanded)
    return str(path.resolve() if path.is_absolute() else (Path(base) / path).resolve())


def _protocol_details(document):
    """Keep the scientific description with the resolved run."""
    details = {key: copy.deepcopy(document[key]) for key in PROTOCOL_DETAILS
               if key in document}
    for key in ("population", "preprocessing", "fitting"):
        if key in details and not isinstance(details[key], dict):
            raise TypeError(f"protocol {key} must be an object")
    if "references" in details:
        references = details["references"]
        if not isinstance(references, list) or not references:
            raise ValueError("protocol references must be a nonempty list")
        for reference in references:
            if (not isinstance(reference, dict)
                    or not all(isinstance(reference.get(key), str)
                               and reference[key].strip()
                               for key in ("title", "role"))
                    or not any(isinstance(reference.get(key), str)
                               and reference[key].strip()
                               for key in ("url", "file"))):
                raise ValueError("each reference needs title, role, and url or file")
    return details


def load_protocol(protocol, *, store=None) -> dict:
    """Load a protocol name, JSON path, or mapping and resolve its paths.

    Inputs use ScanInputs fields. Config is an inline CastConfig or a path
    relative to the repository. Data paths use store/inputs.base_dir.
    Ranking model paths use the repository; proposal names use the run.
    """
    if isinstance(protocol, dict):
        document = copy.deepcopy(protocol)
    else:
        path = Path(protocol)
        if not path.is_file():
            if path.suffix or path.is_absolute() or path.parent != Path("."):
                raise FileNotFoundError(path)
            path = PROTOCOLS / f"{protocol}.json"
        document = _json(path)
        document.setdefault("name", path.stem)
    unknown = set(document) - {"name", "inputs", "config", "ranking", "evaluation", "fitted_dir", *PROTOCOL_DETAILS}
    if unknown:
        raise KeyError(f"unknown protocol fields {sorted(unknown)}")
    details = _protocol_details(document)
    inputs = copy.deepcopy(document["inputs"])
    base = _path(store if store is not None else inputs.get("base_dir", ""), Path.cwd())
    inputs["base_dir"] = base
    fields = {f.name for f in dataclasses.fields(ScanInputs)}
    unknown = set(inputs) - fields
    if unknown:
        raise KeyError(f"unknown ScanInputs fields {sorted(unknown)}")
    for key, value in list(inputs.items()):
        if key not in ("base_dir", "out_dir", "missing_rows") and not key.endswith("_groups"):
            inputs[key] = _path(value, base)
    config = document.get("config", "configs/method.json")
    if isinstance(config, (str, Path)):
        config = _json(_path(config, REPO))
    config = config.get("config", config)
    config = CastConfig.from_dict(config).to_dict()
    ranking = {"ranker": "models/proposal_ranker_avdeepfake1m_external_reference.json",
               "proposals": "proposals_learned_rescored.csv", "fit": False,
               **document.get("ranking", {})}
    ranking_fields = {"ranker", "proposals", "fit", "features", "boundary_cues",
                      "self_splice_manifest", "build_record", "self_splice_proposals",
                      "self_splice_boundary_cues", "clip_gate"}
    if set(ranking) - ranking_fields:
        raise KeyError(f"unknown ranking fields {sorted(set(ranking) - ranking_fields)}")
    ranking["ranker"] = _path(ranking["ranker"], REPO)
    for key in ("boundary_cues", "self_splice_manifest", "build_record", "self_splice_boundary_cues"):
        if ranking.get(key):
            ranking[key] = _path(ranking[key], base)
    evaluation = copy.deepcopy(document.get("evaluation", {}))
    if set(evaluation) - {"labels", "cohort", "manifest", "detection", "localization", "views", "metrics"}:
        raise KeyError("evaluation fields are labels, cohort, manifest, detection, localization, views, metrics")
    if "metrics" in evaluation:
        expected = runner.reported_metrics(
            detection=evaluation.get("detection", True) is not False,
            localization=evaluation.get("localization", True) is not False)
        if evaluation["metrics"] != expected:
            raise ValueError("evaluation.metrics must match the reported metric names")
    if "views" in evaluation:
        expected = (runner.localization_views()
                    if evaluation.get("localization", True) is not False else {})
        if evaluation["views"] != expected:
            raise ValueError("evaluation.views must match the published localization views")
    evaluation["labels"] = _path(evaluation.get("labels", inputs["labels_csv"]), base)
    if evaluation.get("cohort"):
        evaluation["cohort"] = _path(evaluation["cohort"], base)
    if evaluation.get("manifest"):
        evaluation["manifest"] = _path(evaluation["manifest"], base)
    localization = evaluation.get("localization")
    if isinstance(localization, dict) and localization.get("complete_labels"):
        localization["complete_labels"] = _path(localization["complete_labels"], base)
    return {"name": document.get("name", "custom"), **details, "inputs": inputs,
            "config": config, "ranking": ranking, "evaluation": evaluation,
            **({"fitted_dir": _path(document["fitted_dir"], base)}
               if document.get("fitted_dir") else {})}


def _save_protocol(protocol, out):
    runner.write_result(protocol, Path(out) / "protocol.json")


def _figure_ids(cases):
    if cases is None:
        return ()
    document = _json(cases)
    entries = document.get("cases", document) if isinstance(document, dict) else document
    if isinstance(entries, dict):
        entries = list(entries.values())
    return tuple(dict.fromkeys(entry if isinstance(entry, str) else entry["video_id"]
                               for entry in entries))


def _fit_ranker(protocol, out, *, self_splice_manifest=None):
    from .ranking.build import build_self_splice_boundary_cues, fit_proposal_ranker

    ranking = protocol["ranking"]
    manifest = self_splice_manifest or ranking.get("self_splice_manifest")
    if not manifest:
        raise ValueError("ranker fitting needs a genuine self_splice_manifest")
    sources = runner.read_table(manifest)
    if "target" in sources and not sources.target.eq(0).all():
        raise ValueError("self-splice source manifest must contain only genuine clips")
    corpus = protocol["inputs"].get("self_splice_corpus")
    if not corpus:
        raise ValueError("ranker fitting needs inputs.self_splice_corpus")
    proposals = out / ranking.get("self_splice_proposals", "self_splice_proposals_rescored.csv")
    cues = ranking.get("self_splice_boundary_cues")
    if cues is None:
        cues = out / "self_splice_boundary_cues.npz"
        build_self_splice_boundary_cues(proposals, corpus, manifest, cues,
                                        build_record=ranking.get("build_record"))
    ranker = out / "proposal_ranker.json"
    fit_proposal_ranker(proposals, corpus, cues, ranker,
                        features=ranking.get("features"), protocol=protocol["name"])
    ranking["ranker"] = str(ranker)
    return ranker


def train(protocol, out, *, store=None, nproc=None, fit_ranker=False,
          self_splice_manifest=None, figure_cases=None, manifest=None, log=print) -> Path:
    """Run the declared protocol fits and scan; optionally fit its ranker."""
    protocol = load_protocol(protocol, store=store)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "clip_scores.csv").exists():
        raise FileExistsError(f"scan already exists: {out}; evaluate it with --run")
    config = CastConfig.from_dict(protocol["config"])
    if nproc is not None:
        config = dataclasses.replace(config, execution=dataclasses.replace(config.execution, nproc=nproc))
    protocol["config"] = config.to_dict()
    protocol["inputs"]["out_dir"] = str(out)
    if manifest:
        protocol["evaluation"]["manifest"] = str(Path(manifest).resolve())
    inputs = ScanInputs(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in protocol["inputs"].items()})
    _save_protocol(protocol, out)
    runner.write_result(config.to_dict(), out / "config.json")
    runner.write_result(inputs.to_dict(), out / "inputs.json")
    kwargs = {"log": log}
    if protocol.get("fitted_dir"):
        kwargs["fitted_dir"] = protocol["fitted_dir"]
    if figure_cases is not None:
        kwargs["figure_video_ids"] = _figure_ids(figure_cases)
    run_detection(inputs, config, **kwargs)
    if fit_ranker or protocol["ranking"].get("fit"):
        _fit_ranker(protocol, out, self_splice_manifest=self_splice_manifest)
    if manifest:
        # The same declared media manifest can be used when evaluating this run.
        runner.read_table(manifest)
    _save_protocol(protocol, out)
    return out


def _reuse_run(run, out):
    """Reuse ordinary scan artifacts without running fitting or scanning."""
    run, out = Path(run).resolve(), Path(out).resolve()
    if not (run / "clip_scores.csv").is_file():
        raise FileNotFoundError(run / "clip_scores.csv")
    existing = out / "clip_scores.csv"
    if run != out and existing.exists() and existing.resolve() != (run / "clip_scores.csv").resolve():
        raise FileExistsError(f"output already contains another scan: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if run != out:
        generated = {"protocol.json", "config.json", "inputs.json", "detection.json",
                     "localization.json", "localization_complete.json", "metrics.json", "calibration.csv",
                     "proposals_ranked.csv", "boundary_cues.npz", "self_splice_boundary_cues.npz"}
        for source in run.iterdir():
            if source.is_file() and source.suffix in (".csv", ".npz", ".json", ".pkl"):
                if source.name in generated:
                    continue
                target = out / source.name
                if not target.exists():
                    if source.name == "proposal_ranker.json":
                        shutil.copy2(source, target)
                    else:
                        target.symlink_to(source)
        for source in run.glob("fitted*"):
            if source.is_dir() and not (out / source.name).exists():
                (out / source.name).symlink_to(source, target_is_directory=True)
        if (run / "boundary_curves").is_dir() and not (out / "boundary_curves").exists():
            (out / "boundary_curves").symlink_to(run / "boundary_curves", target_is_directory=True)
    return out


def _rank(protocol, run, labels, *, manifest=None, figure_cases=None):
    from .ranking.build import build_boundary_cues, rank_proposals
    from .ranking import ProposalRanker

    ranking = protocol["ranking"]
    ranker_path = Path(ranking["ranker"])
    saved = run / "proposal_ranker.json"
    if not saved.exists():
        shutil.copy2(ranker_path, saved)
    model = ProposalRanker.load(saved)
    cues = ranking.get("boundary_cues")
    if "audio_boundary_cue" in model.feature_names and cues is None:
        if manifest is None:
            raise ValueError("audio boundary cues need --manifest or ranking.boundary_cues")
        cues = run / "boundary_cues.npz"
        if not cues.exists():
            kwargs = {}
            if figure_cases is not None and not (run / "boundary_curves").exists():
                kwargs = {"curves_dir": run / "boundary_curves", "curve_video_ids": _figure_ids(figure_cases)}
            build_boundary_cues(run / ranking["proposals"], run / "clip_scores.csv",
                                labels, manifest, cues, **kwargs)
    ranked = run / "proposals_ranked.csv"
    if ranked.exists():
        return ranked
    # Existing ranker fits are applied as saved; evaluation never refits them.
    rank_proposals(run / ranking["proposals"], run / "clip_scores.csv", labels,
                   saved, cues, ranked)
    gate = ranking.get("clip_gate")
    if gate is not None:
        scores = runner.read_table(run / "clip_scores.csv")
        keep = set(scores.loc[scores.clip_pvalue <= float(gate), "video_id"])
        frame = pd.read_csv(ranked)
        frame[frame.video_id.isin(keep)].to_csv(ranked, index=False)
    protocol["ranking"]["ranker"] = str(saved)
    return ranked


def evaluate(protocol=None, out=None, *, run=None, store=None, nproc=None,
             bootstrap=0, figure_cases=None, manifest=None, log=print) -> Path:
    """Evaluate a new protocol scan or an existing run without refitting."""
    if out is None:
        raise ValueError("evaluation needs an output directory")
    if protocol is None:
        if run is None:
            raise ValueError("provide --protocol or --run")
        protocol = Path(run) / "protocol.json"
    protocol = load_protocol(protocol, store=store)
    if run is None:
        out = train(protocol, out, nproc=nproc, figure_cases=figure_cases, manifest=manifest, log=log)
    else:
        out = _reuse_run(run, out)
    evaluation = protocol["evaluation"]
    labels = evaluation["labels"]
    manifest = manifest or evaluation.get("manifest")
    # Rank over the scan's own metadata order. Cohort selection happens only
    # after the fitted ranker has been applied to the emission.
    if evaluation.get("localization", True) is not False:
        ranked = _rank(protocol, out, protocol["inputs"]["labels_csv"],
                       manifest=manifest, figure_cases=figure_cases)
    else:
        ranked = None
    common = {"cohort": evaluation.get("cohort"), "bootstrap": bootstrap}
    metrics = {"protocol": protocol["name"]}
    if evaluation.get("detection", True) is not False:
        settings = {**(evaluation.get("detection") or {}), **common}
        metrics["detection"] = runner.detection(out / "clip_scores.csv", labels, **settings)
        runner.write_result(metrics["detection"], out / "detection.json")
    if ranked is not None:
        settings = {"unscored": "exclude", **(evaluation.get("localization") or {}), **common}
        complete = settings.pop("complete", False)
        complete_labels = settings.pop("complete_labels", labels)
        metrics["localization"] = runner.localization(ranked, labels, clip_scores=out / "clip_scores.csv", **settings)
        runner.write_result(metrics["localization"], out / "localization.json")
        if complete:
            # Golden complete-validation localization reports point estimates only.
            metrics["localization_complete"] = runner.localization(
                ranked, complete_labels, clip_scores=out / "clip_scores.csv",
                **{**settings, "unscored": "empty", "bootstrap": 0})
            runner.write_result(metrics["localization_complete"], out / "localization_complete.json")
    runner.write_result(metrics, out / "metrics.json")
    columns = pd.read_csv(out / "clip_scores.csv", nrows=0).columns
    if "exact_learned_scan_rank" in columns:
        runner.export_calibration(out / "clip_scores.csv", labels, out / "calibration.csv",
                                  cohort=evaluation.get("cohort"))
    if figure_cases is not None:
        if manifest is None:
            raise ValueError("--figure-cases needs --manifest")
        from .figure_data import export_figure_data
        export_figure_data(out, figure_cases, manifest, labels)
    runner.write_result(protocol["config"], out / "config.json")
    runner.write_result({**protocol["inputs"], "out_dir": str(out)}, out / "inputs.json")
    _save_protocol(protocol, out)
    return out


def arm_definition(definitions, arm):
    for table in ("arms", "variants", "cumulative"):
        if arm in definitions.get(table, {}):
            return copy.deepcopy(definitions[table][arm])
    choices = sorted(name for table in ("arms", "variants", "cumulative")
                     for name in definitions.get(table, {}))
    raise KeyError(f"unknown ablation {arm!r}; expected one of {choices}")


def ablation_protocol(protocol, arm, *, store=None, definitions=None):
    """Apply a published arm's real scan and ranking deltas."""
    definitions = _json(ABLATIONS) if definitions is None else definitions
    protocol = load_protocol(protocol, store=store)
    delta = arm_definition(definitions, arm)
    protocol["inputs"].update(delta.get("inputs", {}))
    for section, values in delta.get("config", {}).items():
        protocol["config"].setdefault(section, {}).update(values)
    protocol["ranking"].update(delta.get("ranking", {}))
    # Each changed scan has its own self-splice emission and ranker fit.
    protocol["ranking"]["fit"] = delta.get("fit_ranker", False)
    protocol["name"] = f"{protocol['name']}/{arm}"
    return load_protocol(protocol), delta.get("reuse_scan", False)


def run_ablation(protocol, arm, out, *, run=None, store=None, nproc=None,
                 bootstrap=0, figure_cases=None, manifest=None,
                 self_splice_manifest=None, log=print) -> Path:
    """Execute one arm, reusing only the emissions declared by that arm."""
    varied, reuse = ablation_protocol(protocol, arm, store=store)
    if self_splice_manifest:
        varied["ranking"]["self_splice_manifest"] = str(Path(self_splice_manifest).resolve())
    if reuse:
        if run is None:
            raise ValueError(f"{arm} reuses the baseline emission; provide --run")
        if Path(run).resolve() == Path(out).resolve():
            raise ValueError("ablation output must differ from the baseline run")
        out = _reuse_run(run, out)
        # Feature refits must not accidentally load the baseline's ranker.
        saved = out / "proposal_ranker.json"
        if saved.exists() and (varied["ranking"]["fit"] or arm == "without_proposal_ranker"):
            saved.unlink()
        if varied["ranking"]["fit"]:
            _fit_ranker(varied, out)
    else:
        if run is not None:
            raise ValueError(f"{arm} requires its own protocol scan; omit --run")
        out = train(varied, out, nproc=nproc, figure_cases=figure_cases, manifest=manifest, log=log)
    return evaluate(varied, out, run=out, bootstrap=bootstrap,
                    figure_cases=figure_cases, manifest=manifest, log=log)
