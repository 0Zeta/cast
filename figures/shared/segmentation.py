"""Display saved retained candidates and the saved penalized decoder selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import _ticks


def segmentation(data: Path, *, top=12) -> dict:
    case = json.loads((data / 'selected_case.json').read_text())
    for name in ('candidates.csv', 'decoded.csv'):
        if not (data / name).is_file():
            raise FileNotFoundError(f'{data}: segmentation needs saved {name}')
    candidates = pd.read_csv(data / 'candidates.csv', float_precision='round_trip')
    decoded = pd.read_csv(data / 'decoded.csv', float_precision='round_trip')
    if candidates.empty:
        raise ValueError(f'{data}: no saved retained candidates')
    if candidates['rank'].duplicated().any() or set(candidates['rank']) != set(range(len(candidates))):
        raise ValueError(f'{data}: candidate ranks must be contiguous from zero')
    candidates = candidates.sort_values('score', ascending=False, kind='stable').head(top)
    # Segmentation displays the complete scored clip, independently of gallery crops.
    start, end = 0., case['duration_s']
    entries = []
    for row in candidates.itertuples():
        if not (np.isfinite(row.score) and 0 <= row.start_s < row.end_s <= case['duration_s']):
            raise ValueError(f'{data}: invalid saved candidate')
        if row.end_s > start and row.start_s < end:
            entries.append({'start': max(start, row.start_s), 'end': min(end, row.end_s),
                            'score': float(row.score), 'rank': int(row.rank)})
    if not entries:
        raise ValueError(f'{data}: no top candidates intersect the selected view')
    scores = [item['score'] for item in entries]
    lo, hi = min(0., min(scores)), max(0., max(scores))
    span = hi - lo or 1.
    return {'case': case.get('id', data.parent.name), 'window': [start, end],
            'candidates': entries, 'top': top,
            'gt': [[g['start_s'], g['end_s']] for g in case['ground_truth_intervals']],
            'decoded': decoded[['start_s', 'end_s', 'score']].to_dict('records'),
            'ymin': lo - .05 * span, 'ymax': hi + .08 * span,
            'yticks': np.linspace(lo, hi if hi > lo else lo + 1., 5).tolist(),
            'ticks': _ticks((start, end))}
