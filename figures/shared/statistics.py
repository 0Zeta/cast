"""Map saved evaluator records to plots; no metric or interval estimation."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _unit(value, field):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{field} must be a finite fraction in [0,1]')
    return value


def localization(paths: list[Path], *, view='benchmark', intervals=False) -> dict:
    panels = [dict(title='Average precision', xlabel='Temporal IoU threshold', series=[]),
              dict(title='Average recall', xlabel='Proposals per clip', series=[])]
    common_grid, common_keys, confidence = None, None, None
    datasets = []
    for path in paths:
        result = json.loads(Path(path).read_text())
        record = result.get('localization')
        if not record:
            raise ValueError(f'{path}: no saved localization metrics')
        if view == 'benchmark':
            point = record.get('point', record.get('overall', {}).get('point'))
            evaluator = record['evaluator']
            uncertainty = record.get('uncertainty', {})
        else:
            point = record.get('views', {}).get(view)
            evaluator = record.get('view_evaluators', {}).get(view)
            uncertainty = record.get('view_uncertainty', {}).get(view, {})
            if point:
                # AP is independent of the recall grid; retain all saved AP thresholds.
                benchmark = record.get('point', record.get('overall', {}).get('point', {}))
                point = {**point, **{k: v for k, v in benchmark.items() if k.startswith('ap@')}}
        if not point or not evaluator:
            raise ValueError(f'{path}: no saved localization {view} view')
        grid = evaluator['ar_iou_grid']
        keys = sorted(key for key in point if key.startswith(('ap@', 'ar@')))
        if common_grid is not None and (grid != common_grid or keys != common_keys):
            raise ValueError('comparison runs need identical saved AP/AR grids and budgets')
        common_grid, common_keys = grid, keys
        marginal = uncertainty.get('marginal', {})
        if intervals:
            if not marginal:
                raise ValueError(f'{path}: confidence plot needs saved marginal intervals')
            level = _unit(uncertainty['confidence_level'], 'confidence_level')
            if confidence is not None and level != confidence:
                raise ValueError('confidence levels differ between runs')
            confidence = level
        protocol_path = Path(path).parent / 'protocol.json'
        protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
        name = result.get('dataset', protocol.get('population', {}).get('dataset',
                          result.get('protocol', Path(path).parent.name)))
        datasets.append(result.get('dataset', protocol.get('population', {}).get('dataset')))
        for panel, prefix in zip(panels, ('ap@', 'ar@'), strict=True):
            points = []
            for key in sorted((key for key in keys if key.startswith(prefix)),
                              key=lambda key: float(key.split('@')[1])):
                x = float(key.split('@')[1])
                y = _unit(point[key], key)
                item = {'x': x, 'y': y, 'lo': None, 'hi': None}
                if intervals:
                    if key not in marginal:
                        raise ValueError(f'{path}: missing saved interval for {key}')
                    item['lo'] = _unit(marginal[key]['lo'], key + '.lo')
                    item['hi'] = _unit(marginal[key]['hi'], key + '.hi')
                    if item['lo'] > item['hi'] or not math.isclose(float(marginal[key]['point']), y, abs_tol=1e-12):
                        raise ValueError(f'{path}: inconsistent saved interval for {key}')
                points.append(item)
            if not points:
                raise ValueError(f'{path}: no saved {prefix} metrics')
            panel['series'].append({'name': ('CAST (' + name + ')') if view == 'comparison' else name,
                                    'points': points, 'markers': view == 'comparison'})
    citation = None
    if view == 'comparison' and all(dataset == 'AV-Deepfake1M' for dataset in datasets):
        comparator = json.loads((Path(__file__).resolve().parents[1] /
                                  'fig3-statistical-plots/gem_tfl_avdeepfake1m.json').read_text())
        if not _same_grid(common_grid, comparator['ar_iou_grid']):
            raise ValueError('GEM-TFL comparison requires its saved 0.1:0.1:0.7 recall grid')
        for panel, key in zip(panels, ('ap', 'ar'), strict=True):
            saved_x = {p['x'] for p in panel['series'][0]['points']}
            if not {x for x, _ in comparator[key]} <= saved_x:
                raise ValueError(f'GEM-TFL comparison needs all quoted {key} reporting points')
            panel['series'].append({'name': 'GEM-TFL (Zhu et al., 2026)', 'dashed': True, 'markers': True,
                                    'points': [{'x': x, 'y': y, 'lo': None, 'hi': None}
                                               for x, y in comparator[key]]})
        citation = comparator['citation']
    for panel in panels:
        xs = [p['x'] for p in panel['series'][0]['points']]
        ticks = []
        for x in xs:
            if not ticks or x - ticks[-1] >= max(xs) * .15:
                ticks.append(x)
        panel.update(xmin=0., xmax=max(xs) * 1.02, xticks=ticks, ymin=0., ymax=1.,
                     yticks=[0., .2, .4, .6, .8, 1.], diagonal=False, xpercent=False)
    return {'panels': panels, 'confidence': confidence, 'citation': citation,
            'note': ('AR uses saved IoU grid: ' + ', '.join(f'{x:g}' for x in common_grid)
                     + ('. GEM-TFL: Zhu et al. (2026), Table 2.' if citation else ''))}


def _same_grid(left, right):
    return len(left) == len(right) and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(left, right))
