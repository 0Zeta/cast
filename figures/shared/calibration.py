"""Render complete-scan calibration counts with pointwise Wilson intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    point = k / n
    denominator = 1.0 + z * z / n
    center = point + z * z / (2.0 * n)
    half = z * np.sqrt(point * (1.0 - point) / n + z * z / (4.0 * n * n))
    return (center - half) / denominator, (center + half) / denominator


def load_curves(runs: list[Path], paper_counts: Path) -> list[dict]:
    tables = []
    if runs:
        for run in runs:
            path = run / 'calibration.csv'
            protocol_path = run / 'protocol.json'
            protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
            label = protocol.get('population', {}).get('dataset', protocol.get('name', run.name))
            tables.append((label, path, pd.read_csv(path, float_precision='round_trip')))
    else:
        saved = pd.read_csv(paper_counts, float_precision='round_trip')
        if 'dataset' not in saved or saved.dataset.isna().any():
            raise ValueError(f'{paper_counts}: expected dataset labels')
        tables = [(label, paper_counts, table) for label, table in saved.groupby('dataset', sort=False)]
    curves = []
    for label, path, table in tables:
        required = ['alpha', 'rejections', 'n_genuine', 'rate']
        if not set(required) <= set(table):
            raise ValueError(f'{path}: expected {required}')
        if table.empty or not np.isfinite(table[required].to_numpy(dtype=float)).all():
            raise ValueError(f'{path}: empty or non-finite calibration data')
        counts = table[['rejections', 'n_genuine']].to_numpy()
        if ((counts != np.floor(counts)).any() or (table.n_genuine <= 0).any() or
                table.n_genuine.nunique() != 1 or (table.rejections < 0).any() or
                (table.rejections > table.n_genuine).any()):
            raise ValueError(f'{path}: calibration needs valid integer counts from one genuine population')
        if (not table.rejections.is_monotonic_increasing or
                not np.allclose(table.rate, table.rejections / table.n_genuine, rtol=0, atol=1e-12)):
            raise ValueError(f'{path}: rates disagree with counts')
        if (table.alpha.duplicated().any() or not table.alpha.is_monotonic_increasing or
                not table.alpha.between(0, 1).all() or not table.rate.between(0, 1).all()):
            raise ValueError(f'{path}: alpha/rate must lie in [0,1] in strict alpha order')
        curves.append({'name': label, 'points': [
            {'alpha': float(row.alpha), 'rejections': int(row.rejections), 'n_genuine': int(row.n_genuine)}
            for row in table.itertuples()]})
    if not curves:
        raise ValueError(f'{paper_counts}: no calibration curves')
    return curves


def render_calibration(curves: list[dict], pdf: Path, *, figures: Path, profile: dict) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    for font in sorted((figures / 'shared/fonts').glob('NimbusRoman-*.otf')):
        font_manager.fontManager.addfont(font)
    ink, soft, line = '#263746', '#6F7C87', '#BCC6CE'
    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 8.5,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelcolor': ink,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'xtick.color': soft,
        'ytick.color': soft,
    }):
        figure, axis = plt.subplots(figsize=(3.35, profile['height_inches']), facecolor='white')
        try:
            figure.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=profile['bottom'])
            axis.plot([0.0, 0.105], [0.0, 0.105], color=line,
                      linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
            palette = ('#0072B2', '#D55E00', '#4C6FAF', ink)
            for index, curve in enumerate(curves):
                color = palette[index % len(palette)]
                points = curve['points']
                alpha = np.array([p['alpha'] for p in points])
                rates = np.array([p['rejections'] / p['n_genuine'] for p in points])
                lower, upper = np.array([wilson(p['rejections'], p['n_genuine']) for p in points]).T
                axis.fill_between(alpha, lower, upper, color=color, alpha=0.20,
                                  linewidth=0, zorder=0.5)
                axis.plot(alpha, rates, color=color, linestyle='-',
                          linewidth=1.55 if index == 0 else 1.35,
                          label=curve['name'], zorder=4 if index == 0 else 2)
            axis.set_xlim(*profile['xlim'])
            axis.set_ylim(*profile['ylim'])
            axis.set_xticks(profile['nominal_ticks'])
            axis.set_yticks(profile['rate_ticks'])
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.set_xlabel(r'Nominal level $\alpha$', fontsize=8.3, labelpad=4)
            axis.set_ylabel(profile['title'], fontsize=8.3, labelpad=5)
            axis.legend(loc='upper left', frameon=False, fontsize=7.0,
                        handlelength=2.6, handletextpad=0.6, labelspacing=0.45, borderaxespad=0.15)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            for side in ('bottom', 'left'):
                axis.spines[side].set_color(line)
                axis.spines[side].set_linewidth(0.65)
            axis.tick_params(axis='both', colors=soft, width=0.6, length=2.8, labelsize=7.4)
            pdf.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(pdf, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.01)
        finally:
            plt.close(figure)
    return pdf
