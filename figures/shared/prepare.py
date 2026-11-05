"""Prepare the shared CeTZ panels from already computed scientific values.

The coordinate maps retain the accepted qualitative v2 layout: no smoothing,
restandardization, or fitting. Media preparation uses ffmpeg and Pillow.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP = 160
DISPLAY_PADDING = 0.06

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fraction(value: float, window: tuple[float, float]) -> float:
    return float(np.clip((value - window[0]) / (window[1] - window[0]), 0, 1))


def _display_range(values: np.ndarray, baseline: float) -> tuple[float, float]:
    lo, hi = min(float(values.min()), baseline), max(float(values.max()), baseline)
    # A constant trace at its baseline still needs a nonzero affine map.
    span = hi - lo
    padding = DISPLAY_PADDING * (span if span > 0 else max(1.0, abs(baseline)))
    return lo - padding, hi + padding


def _plot_samples(times: np.ndarray, values: np.ndarray,
                  window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Clip the existing polyline, interpolating only where the crop cuts it."""
    lo, hi = max(window[0], float(times[0])), min(window[1], float(times[-1]))
    if lo > hi:
        return np.empty(0), np.empty(0)
    inner = times[(times > lo) & (times < hi)]
    plotted_t = np.unique(np.concatenate(([lo], inner, [hi])))
    return plotted_t, np.interp(plotted_t, times, values)


def _track(times: np.ndarray, values: np.ndarray, window: tuple[float, float],
           limits: tuple[float, float], baseline: float | None, direction: str) -> dict:
    plotted_t, plotted_v = _plot_samples(times, values, window)
    lo, hi = limits
    normalized = np.clip((plotted_v - lo) / (hi - lo), 0, 1)
    original = (times >= window[0]) & (times <= window[1])
    return {
        "points": [[_fraction(float(t), window), float(y)]
                   for t, y in zip(plotted_t, normalized, strict=True)],
        "baseline": None if baseline is None else float((baseline - lo) / (hi - lo)),
        "direction": direction, "display_min": lo, "display_max": hi,
        "full_clip_min": float(values.min()), "full_clip_max": float(values.max()),
        "plotted_samples": np.column_stack((plotted_t, plotted_v)).tolist(),
        "original_samples": np.column_stack((times[original], values[original])).tolist(),
    }


def _save_spectrogram(path: Path, log_power: np.ndarray,
                      window: tuple[float, float]) -> dict:
    """Uniform time columns; high energy is dark ink, without overlays."""
    times = np.arange(len(log_power)) * HOP / SAMPLE_RATE
    width = max(1, math.ceil((window[1] - window[0]) * SAMPLE_RATE / HOP))
    column_times = window[0] + (np.arange(width) + 0.5) * (window[1] - window[0]) / width
    shown = np.asarray([np.interp(column_times, times, column) for column in log_power.T])
    lo, hi = map(float, np.quantile(log_power, (0.05, 0.995)))
    ink = np.zeros_like(shown) if hi == lo else np.clip((shown - lo) / (hi - lo), 0, 1)
    pixels = np.rint(255 * (1 - ink[::-1])).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return {"frequency_hz": [0, SAMPLE_RATE / 2], "high_frequency_at_top": True,
            "mode": "L", "high_energy": "dark", "log_power_min": lo,
            "log_power_max": hi, "contrast_quantiles_full_clip": [0.05, 0.995],
            "size_px": [width, len(log_power[0])],
            "time_interpolation": "linear at uniform column centers; hold at clip edges"}


def _ticks(window: tuple[float, float]) -> list[dict]:
    target = (window[1] - window[0]) / 4
    decade = 10.0 ** math.floor(math.log10(target))
    step = next(multiplier * decade for multiplier in (1, 2, 5, 10)
                if multiplier * decade >= target)
    first = math.ceil(window[0] / step - 1e-10)
    last = math.floor(window[1] / step + 1e-10)
    decimals = max(1, -math.floor(math.log10(step)))
    ticks = []
    for index in range(first, last + 1):
        value = round(index * step, decimals)
        ticks.append({"x": _fraction(value, window), "label": f"{value:.{decimals}f} s"})
    return ticks


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def _frame(source: Path, time_s: float, output: Path) -> None:
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', repr(time_s), '-i', str(source),
                    '-frames:v', '1', '-y', str(output)], check=True, capture_output=True)
    if not output.is_file():
        raise ValueError(f'no media frame at {time_s}: {source}')


def prepare_case(data: Path, output: Path, *, window=None, selected_tracks=None,
                 frame_layout=None, media=None) -> dict:
    """Prepare a whole panel once for both static and video rendering."""
    data, output = Path(data).resolve(), Path(output).resolve()
    _require(not output.is_relative_to(data) and not data.is_relative_to(output),
             'output must be separate from scientific inputs')
    case = json.loads((data / 'selected_case.json').read_text())
    _require(math.isfinite(case['fps']) and case['fps'] > 0, 'invalid case fps')
    origin = float(case.get('media_time_origin_s', 0.0))
    rows = _read_csv(data / 'tracks.csv')
    if not rows:
        raise ValueError(f'empty tracks: {data}')
    times = np.array([float(row['time_s']) for row in rows])
    _require(np.array_equal(np.array([int(r['frame']) for r in rows]),
                            np.arange(case['n_frames'])), 'noncontiguous frame indices')
    _require(np.isfinite(times).all() and (np.diff(times) > 0).all(), 'invalid track times')
    _require(np.allclose(times, np.arange(len(times)) / case['fps'], rtol=0, atol=1e-7),
             'track times differ from frame timestamps')
    gt = [(float(item['start_s']), float(item['end_s']))
          for item in case.get('ground_truth_intervals', [])]
    if not gt and case.get('ground_truth'):
        gt = [(case['ground_truth']['start_s'], case['ground_truth']['end_s'])]
    if frame_layout is not None:
        _require(len(gt) == 1 and len(frame_layout) == 3 and sum(frame_layout) == 9,
                 'nine-bin frame layout needs one GT interval')
        before, forged, after = frame_layout
        _require(forged > 0 and min(frame_layout) >= 0, 'invalid frame layout')
        step = (gt[0][1] - gt[0][0]) / forged
        window = [gt[0][0] - before * step, gt[0][1] + after * step]
    if window is None:
        if case.get('crop'):
            window = case['crop']
        elif case.get('display_window_s'):
            window = [case['display_window_s']['start'], case['display_window_s']['end']]
        else:
            window = [0.0, case['duration_s']]
    window = tuple(map(float, window))
    _require(0 <= window[0] < window[1] <= case['duration_s'], 'window outside clip')
    groups = case.get('score_groups', [key for key in rows[0] if key not in ('frame', 'time_s')])
    displays = {}
    for key in groups:
        values = np.array([float(row[key]) for row in rows])
        _require(np.isfinite(values).all(), f'non-finite track: {key}')
        baseline = 1.0 if key == 'sync' else 0.0
        displays[key] = _track(times, values, window, _display_range(values, baseline),
                               None if key == 'fused' else baseline,
                               'up' if key in ('sync', 'fused') else 'down')
    curve_path = data / 'boundary_curve.csv'
    if curve_path.exists():
        curve = _read_csv(curve_path)
        flux_t = np.array([float(row['time_s']) for row in curve])
        flux = np.array([float(row['flux']) for row in curve])
        _require(len(flux_t) > 1 and np.isfinite(flux_t).all()
                 and (np.diff(flux_t) > 0).all() and np.isfinite(flux).all()
                 and ((0 <= flux) & (flux <= 1)).all(), 'invalid saved boundary curve')
        # Saved curves use the untrimmed waveform clock; panels use score time.
        displays['flux'] = _track(flux_t - origin, flux, window, (0.0, 1.0), None, 'up')
    flux_peaks = []
    if (data / 'boundary_peaks.csv').is_file():
        for peak in _read_csv(data / 'boundary_peaks.csv'):
            time_s, value = float(peak['time_s']) - origin, float(peak['flux'])
            _require('flux' in displays and math.isfinite(time_s) and 0 <= value <= 1,
                     'invalid saved boundary marker')
            if int(peak['rank']) == 0 and window[0] <= time_s <= window[1]:
                flux_peaks.append([_fraction(time_s, window), value])
    decoder_path = data / 'decoder_tracks.csv'
    if decoder_path.is_file():
        decoder = _read_csv(decoder_path)
        _require(len(decoder) == len(times), 'decoder frame count differs from scientific tracks')
        decoder_t = np.array([float(row['time_s']) for row in decoder])
        _require(np.allclose(times, decoder_t, rtol=0, atol=1e-7), 'decoder times differ')
        values = np.array([float(row['learned_marginal_inclusion']) for row in decoder])
        _require(np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(),
                 'invalid saved marginal inclusion')
        displays['marginal'] = _track(times, values, window, (0., 1.), None, 'up')
    selected = selected_tracks if selected_tracks is not None else case.get('selected_tracks', list(displays))
    _require(bool(selected) and not set(selected) - set(displays),
             f'{data.parent.name}: requested absent scientific tracks: {sorted(set(selected) - set(displays))}')
    proposals = []
    for row in _read_csv(data / 'proposals.csv'):
        start, end = float(row['start_s']), float(row['end_s'])
        start_f, end_f = int(row['start_f']), int(row['end_f'])
        _require(math.isclose(start, start_f / case['fps'], abs_tol=1e-7)
                 and math.isclose(end, (end_f + 1) / case['fps'], abs_tol=1e-7),
                 'proposal times disagree with inclusive frames')
        proposals.append({'rank': int(row['rank']) + 1, 'start_s': start, 'end_s': end,
                          'start_f': start_f, 'end_f': end_f, 'iou': float(row['iou']),
                          'score': float(row['score']), 'start': _fraction(start, window),
                          'end': _fraction(end, window)})
    decoded = None
    if (data / 'decoded.csv').is_file():
        decoded = []
        for row in _read_csv(data / 'decoded.csv'):
            a, b = float(row['start_s']), float(row['end_s'])
            _require(0 <= a < b <= case['duration_s'], 'decoded interval outside clip')
            _require(math.isclose(a, int(row['start_f']) / case['fps'], abs_tol=1e-7)
                     and math.isclose(b, (int(row['end_f']) + 1) / case['fps'], abs_tol=1e-7),
                     'decoded times disagree with inclusive frames')
            if b > window[0] and a < window[1]:
                decoded.append([_fraction(a, window), _fraction(b, window)])
    media = dict(case.get('media', {}) if media is None else media)
    local_source = data / 'source.mp4'
    if not media and local_source.is_file():
        media = {'video': str(local_source), 'audio': str(local_source)}
    output.mkdir(parents=True, exist_ok=True)
    frames, frame_manifest = [], []
    if 'video' in media:
        source = Path(media['video'])
        _require(source.is_file(), f'missing requested video: {source}')
        centers = media.get('frame_times_s', [window[0] + (i + .5) / 9 * (window[1] - window[0]) for i in range(9)])
        _require(len(centers) == 9 and all(window[0] <= t <= window[1] for t in centers),
                 'frame_times_s needs nine times inside the view')
        for index, time_s in enumerate(centers):
            frame_index = min(int(round(time_s * case['fps'])), case['n_frames'] - 1)
            media_time = origin + frame_index / case['fps']
            name = f'frame_{index:02d}.png'
            _frame(source, media_time, output / name)
            frames.append(name)
            frame_manifest.append({'frame': frame_index, 'time_s': time_s,
                                   'media_time_s': media_time, 'path': name})
    if 'audio' in media:
        source = Path(media['audio'])
        _require(source.is_file(), f'missing requested audio: {source}')
        result = subprocess.run(['ffmpeg', '-v', 'error', '-ss', repr(origin), '-i', str(source),
                                 '-t', repr(case['duration_s']), '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                                 '-f', 'f32le', '-'], check=True, capture_output=True)
        audio = np.frombuffer(result.stdout, dtype='<f4')
        _require(len(audio) >= FFT_SIZE, 'audio shorter than a spectrogram window')
        windows = np.lib.stride_tricks.sliding_window_view(audio, FFT_SIZE)[::HOP]
        log_power = np.log(np.abs(np.fft.rfft(windows * np.hanning(FFT_SIZE), axis=1)) ** 2 + 1e-12)
        _save_spectrogram(output / 'spectrogram.png', log_power, window)
    panel = {'schema_version': 1, 'case_name': case.get('id', data.parent.name), 'case': case,
             'window': list(window), 'gt': [[_fraction(a, window), _fraction(b, window)]
                                         for a, b in gt if b > window[0] and a < window[1]],
             'modalities': {key: bool(case.get('modality_labels', {}).get(key))
                            for key in ('audio', 'video')},
             'frames': frames, 'spectrogram': 'spectrogram.png', 'tracks': displays,
             'proposals': proposals, 'flux_peaks': flux_peaks, 'ticks': _ticks(window),
             'decoded': decoded,
             'media': [key for key in ('video', 'audio') if key in media],
             'selected_tracks': selected, 'media_sources': media,
             'frame_manifest': frame_manifest}
    (output / 'panel.json').write_text(json.dumps(panel, indent=2, allow_nan=False) + '\n')
    return panel
