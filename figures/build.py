#!/usr/bin/env python3
"""Build standalone CAST figures from saved runs or local C22 case packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.prepare import prepare_case
from shared.calibration import load_curves, render_calibration
from shared.render import gallery, rasterize, render_panel, render_panels, render_source, style
from shared.statistics import localization
from shared.segmentation import segmentation
from shared.video import render_video, still_video

HERE = Path(__file__).resolve().parent
CASE_INDEX = HERE / 'shared/cases/index.json'
PAPER_FOLDERS = {
    'teaser': 'fig1-teaser',
    'overview': 'fig2-method-overview',
    'calibration': 'fig3-statistical-plots',
    'qualitative-main': 'fig4-qualitative-examples',
    'confidence': 'fig3-statistical-plots',
    'segmentation': 'figS2-segmentation-decode',
    'comparison': 'fig3-statistical-plots',
}
TARGETS = ('all', 'teaser', 'overview', 'calibration', 'qualitative', 'qualitative-main',
           'supplementary', 'gallery', 'localization', 'confidence', 'comparison', 'segmentation')
DEFAULT_TARGETS = ('teaser', 'overview', 'calibration', 'qualitative', 'supplementary')
SCORE_ORDER = ('cos', 'ja_B', 'jv_B', 'sync', 'vart_C', 'aart_C', 'flux')


def _case_paths(run: Path | None, index: Path | None = None) -> list[Path]:
    if index is None:
        if run is None:
            raise ValueError('case figures need --run or --cases')
        index = run / 'figure_data/cases.json'
    index = Path(index).resolve()
    cases = json.loads(index.read_text())['cases']
    if not cases:
        raise ValueError(f'{index}: no selected cases')
    return [(index.parent / entry['data']).resolve() for entry in cases]


def _formats(formats: str) -> set[str]:
    value = set(formats.split(','))
    if not value or value - {'pdf', 'png', 'mp4'}:
        raise ValueError('formats must select pdf,png,mp4')
    return value


def _static_outputs(source: Path, target: str, output: Path, build: Path,
                    formats: set[str], *, root: Path, inputs=None) -> list[Path]:
    pdf = render_source(source, output / f'{target}.pdf', root=root, inputs=inputs)
    return _pdf_outputs(pdf, formats)


def _pdf_outputs(pdf: Path, formats: set[str], *, panel=None, geometry=None) -> list[Path]:
    results = [pdf] if 'pdf' in formats else []
    if formats & {'png', 'mp4'}:
        png = pdf.with_suffix('.png')
        rasterize(pdf, png)
        if 'png' in formats:
            results.append(png)
        if 'mp4' in formats:
            video = pdf.with_suffix('.mp4')
            if panel is None:
                still_video(png, video)
            else:
                render_video(panel, png, geometry, video)
            results.append(video)
    return results


def _calibration(runs: list[Path], output: Path) -> Path:
    profile = json.loads((HERE / PAPER_FOLDERS['calibration'] / 'profiles.json').read_text())['calibration']
    curves = load_curves(runs, HERE / PAPER_FOLDERS['calibration'] / 'calibration.csv')
    return render_calibration(curves, output / 'calibration.pdf', figures=HERE, profile=profile)


def _plot_source(spec: dict, build: Path, name: str) -> Path:
    spec_path = build / f'{name}.json'
    spec_path.write_text(json.dumps({**style(HERE), **spec}, allow_nan=False))
    source = build / f'{name}.typ'
    source.write_text((HERE / 'fig3-statistical-plots/figure.typ').read_text().replace(
        'sys.inputs.at("spec")', json.dumps(spec_path.name)))
    return source


def _media(data: Path, root: Path | None) -> dict | None:
    if root is None:
        return None
    # Explicit local media only; there is no external dataset discovery.
    case = json.loads((data / 'selected_case.json').read_text())
    source = root / case.get('id', data.parent.name) / 'data/source.mp4'
    if not source.is_file():
        raise FileNotFoundError(f'matching local media required: {source}')
    return {'video': str(source.resolve()), 'audio': str(source.resolve())}


def _main_qualitative(paths: list[Path], build: Path, output: Path, formats: set[str],
                      media_root: Path | None, profile: Path) -> list[Path]:
    by_id = {json.loads((path / 'selected_case.json').read_text()).get('id', path.parent.name): path
             for path in paths}
    config = json.loads(profile.read_text())
    entries = []
    for view in config['panels']:
        if view['case'] not in by_id:
            raise ValueError(f"main qualitative needs selected case {view['case']}")
        data = by_id[view['case']]
        assets = build / 'qualitative-main' / view['case']
        panel = prepare_case(data, assets, frame_layout=view['frame_layout'],
                             selected_tracks=view['scores'], media=_media(data, media_root))
        absent = set(view['media']) - set(panel['media'])
        if absent:
            raise ValueError(f"{view['case']}: accepted main layout needs local media {sorted(absent)}")
        entries.append({**{k: v for k, v in view.items() if k not in ('case', 'frame_layout')},
                        'data': panel, 'assets': assets})
    pdf, _ = render_panels(HERE, entries, output / 'qualitative-main.pdf',
                           build=build / 'qualitative-main', style_overrides=config.get('style'))
    results = _pdf_outputs(pdf, formats - {'mp4'})
    if 'mp4' in formats:
        # Independent clip clocks and audio each get their own playback file.
        for entry in entries:
            single = output / 'qualitative-main' / f"{entry['data']['case_name']}.pdf"
            pdf, geometry = render_panels(HERE, [entry], single,
                                         build=build / 'qualitative-main', style_overrides=config.get('style'))
            results += _pdf_outputs(pdf, {'mp4'}, panel=entry['data'], geometry=geometry)
    return results


def build_figures(target: str, run: Path | None = None, formats=('pdf',), *, calibration_runs=(),
                  cases=None, media_root=None, metrics=None, comparison_runs=(), output=None,
                  profile=None) -> list[Path]:
    """Render explicit targets from saved scientific values, without refitting."""
    if target not in TARGETS:
        raise ValueError(f'unknown target: {target}')
    run = Path(run).resolve() if run is not None else None
    formats = set(formats)
    if not formats or formats - {'pdf', 'png', 'mp4'}:
        raise ValueError('formats must select pdf,png,mp4')
    output = Path(output).resolve() if output else (run / 'figures' if run else HERE.parent / 'outputs/figures')
    build = run / '.figure_build' if run and output == run / 'figures' else output / '.build'
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    media_root = Path(media_root).resolve() if media_root else None
    metrics = Path(metrics).resolve() if metrics else (run / 'metrics.json' if run else None)
    targets = list(DEFAULT_TARGETS) if target == 'all' else [target]
    if target == 'all' and metrics and metrics.is_file():
        record = json.loads(metrics.read_text()).get('localization', {})
        if record:
            targets.append('localization')
        if record.get('uncertainty', {}).get('marginal'):
            targets.append('confidence')
        if record.get('views', {}).get('comparison'):
            targets.append('comparison')
    if target == 'all':
        paths = _case_paths(run, cases)
        if all((path / 'decoded.csv').is_file() and (path / 'candidates.csv').is_file() for path in paths):
            targets.append('segmentation')
    results = []
    for name in targets:
        if name in ('teaser', 'overview'):
            results += _static_outputs(HERE / PAPER_FOLDERS[name] / 'figure.typ', name, output,
                                       build, formats, root=HERE)
        elif name == 'calibration':
            runs = ([run] if run is not None else []) + list(map(Path, calibration_runs))
            results += _pdf_outputs(_calibration(runs, output), formats)
        elif name in ('localization', 'confidence', 'comparison'):
            if metrics is None:
                raise ValueError(f'{name} needs --run or --metrics')
            sources = [metrics, *(Path(path) / 'metrics.json' for path in comparison_runs)]
            plot_profile = ({'view': 'benchmark', 'intervals': False} if name == 'localization' else
                            json.loads((HERE / PAPER_FOLDERS[name] / 'profiles.json').read_text())[name])
            spec = localization(sources, view=plot_profile['view'], intervals=plot_profile['intervals'])
            source = _plot_source(spec, build, name)
            results += _static_outputs(source, name, output, build, formats, root=build)
        else:
            index = CASE_INDEX if name == 'gallery' and cases is None else cases
            paths = _case_paths(run, index)
            if name == 'qualitative-main':
                results += _main_qualitative(paths, build, output, formats, media_root,
                                             Path(profile) if profile else HERE / PAPER_FOLDERS[name] / 'main.json')
                continue
            pdfs = []
            for data in paths:
                case_name = data.parent.name
                if name == 'segmentation':
                    spec = {**style(HERE), **segmentation(data)}
                    spec_path = build / f'segmentation-{case_name}.json'
                    spec_path.write_text(json.dumps(spec, allow_nan=False))
                    source = build / f'segmentation-{case_name}.typ'
                    source.write_text((HERE / PAPER_FOLDERS[name] / 'figure.typ').read_text().replace(
                        'sys.inputs.at("spec")', json.dumps(spec_path.name)))
                    pdf = render_source(source, output / name / f'{case_name}.pdf', root=build)
                    pdfs.append(pdf)
                    results += _pdf_outputs(pdf, formats)
                    continue
                assets = build / name / case_name
                panel = prepare_case(data, assets, media=_media(data, media_root))
                scores = (panel['selected_tracks'] if name == 'qualitative' else
                          [key for key in SCORE_ORDER if key in panel['tracks']])
                pdf, geometry = render_panel(HERE, panel, assets,
                    output / name / f'{case_name}.pdf',
                    proposals=1 if name == 'qualitative' else 3, scores=scores,
                    decoded=name == 'segmentation')
                pdfs.append(pdf)
                results += _pdf_outputs(pdf, formats, panel=panel, geometry=geometry)
            if 'pdf' in formats:
                combined = output / f'{name}.pdf'
                gallery(pdfs, combined)
                results.append(combined)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('build')
    command.add_argument('target', choices=TARGETS)
    command.add_argument('--run', type=Path)
    command.add_argument('--output', type=Path)
    command.add_argument('--formats', default='pdf')
    command.add_argument('--cases', type=Path, help='local cases.json package index')
    command.add_argument('--profile', type=Path, help='main qualitative view configuration')
    command.add_argument('--media-root', type=Path, help='local root with <case>/data/source.mp4')
    command.add_argument('--metrics', type=Path, help='saved metrics.json; defaults to the run')
    command.add_argument('--calibration-run', type=Path, action='append', default=[])
    command.add_argument('--comparison-run', type=Path, action='append', default=[])
    args = parser.parse_args()
    try:
        paths = build_figures(args.target, args.run, _formats(args.formats),
            calibration_runs=args.calibration_run, comparison_runs=args.comparison_run,
            cases=args.cases, media_root=args.media_root, metrics=args.metrics,
            output=args.output, profile=args.profile)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
