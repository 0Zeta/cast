# Figures

Figures read saved scores, proposals, boundary curves, and evaluation results.
Run these commands from the repository root.

The paper renderer uses Matplotlib 3.10.7 with FreeType 2.13.2.
Install a C++ compiler, Python development headers, `pkg-config`, and FreeType development headers.
Build Matplotlib against system FreeType, then install the renderer:

```sh
python -m pip install \
  --no-binary matplotlib \
  --config-settings setup-args=-Dsystem-freetype=true \
  matplotlib==3.10.7
python -m pip install \
  -e '.[figures]'
```

Install `ffmpeg` and `ffprobe` for media decoding and video output.
Calibration uses Matplotlib; other figures use Typst and CeTZ 0.5.2.
PDF and video output also use pypdfium2, Pillow, and the bundled fonts.
The first build can download Typst packages into its package cache.

## Folder layout

Folders group figures by type. CLI targets use content names such as
`teaser` and `overview`. Each type keeps its layout, components, and styles
together.

Typst files define the drawing and layout.
JSON files hold variable profiles, case selections, or fixed reference data.
The builder prepares run-specific input JSON in the output directory.
Fixed illustrations keep their settings in Typst.

| Folder | Contents | Paper figures |
| --- | --- | --- |
| [fig1-teaser/](fig1-teaser/) | `figure.typ`, `components.typ`, `style.typ`, `figure.pdf` | Fig. 1 |
| [fig2-method-overview/](fig2-method-overview/) | `figure.typ`, `components.typ`, `panels.typ`, `style.typ`, `figure.pdf` | Fig. 2 |
| [fig3-statistical-plots/](fig3-statistical-plots/) | `figure.typ`, `profiles.json`, `calibration.csv`, `gem_tfl_avdeepfake1m.json` | Fig. 3, S1, S3 |
| [fig4-qualitative-examples/](fig4-qualitative-examples/) | `figure.typ`, `main.json`, `cases/<logical-case-id>/` | Fig. 4, S4-S10 |
| [figS2-segmentation-decode/](figS2-segmentation-decode/) | `figure.typ` | Fig. S2 |
| [shared/](shared/README.md) | Common styles, assets, fonts, case lists, and Python helpers | All types |

[profiles.json](fig3-statistical-plots/profiles.json) contains `calibration`
(Fig. 3), `confidence` (Fig. S1), and `comparison` (Fig. S3).
Each profile records its `paper_figure` label.
Calibration uses `shared/calibration.py`.
The other statistical profiles use `fig3-statistical-plots/figure.typ`.

[main.json](fig4-qualitative-examples/main.json) selects the main qualitative
panels and records their `paper_figure` label. Each stored case under
`fig4-qualitative-examples/cases/` contains
`data/`, `qualitative_scores.pdf`, `qualitative_scores_video.mp4`, and
`video_timing.json`.
[shared/cases/index.json](shared/cases/index.json) lists cases in Fig. S4-S10
order, with logical `id`, `paper_figure`, and relative `data` paths.
The builder loops over cases using `fig4-qualitative-examples/figure.typ`
for main and supplementary views. Add variants through profiles and case
selections.

[shared/](shared/README.md) contains reusable support:

| Path | Contents |
| --- | --- |
| `shared/styles/` | `illustration.typ` and `tokens.toml` for common styles |
| `shared/assets/` | Paper frames, measured waveform, frame strip, input selection, and interval-set glyph |
| `shared/cases/` | `paper.json`, `example.json`, and the stored-case `index.json` |
| `shared/fonts/` | Fonts and their licenses |
| `shared/*.py` | Data and media preparation, rendering, video, statistical plots, and illustration asset preparation |

Figures 1 and 2 use the paper's original clip
`video5_vox_celeb_2_id06970_3E8IoUP9kZI_00016`.
[illustration_inputs.json](shared/assets/illustration_inputs.json) records the
frame selection and waveform interval. The shared assets contain frames
30, 32, ..., 52 at 25 Hz and the measured 144-bar waveform from 1.20 to 2.16 s.
The teaser uses frames 30, 38, 44, 48, and 52.
`shared/prepare_illustrations.py` rebuilds the vertical strip from those frames.

## Build targets

| Target | Contents |
| --- | --- |
| `teaser`, `overview` | Illustrative teaser and six-score-group method schematic |
| `calibration` | Genuine false-alarm curves with pointwise 95% Wilson intervals |
| `qualitative` | Selected tracks and the top proposal for each case |
| `qualitative-main` | Two-panel paper layout defined in [main.json](fig4-qualitative-examples/main.json) |
| `supplementary`, `gallery` | Available tracks and the first three proposals |
| `localization` | Saved benchmark AP and AR curves |
| `confidence` | AP and AR curves with saved confidence intervals |
| `comparison` | GEM-TFL reporting grid and the cited AV-Deepfake1M comparator |
| `segmentation` | Retained candidates, ground truth, and the saved penalized decode |
| `all` | Standard illustrations, calibration, case panels, and available metric/decoder plots |

Build figures from a run created with `--figure-cases`:

```sh
python figures/build.py build all \
  --run runs/av1m \
  --formats pdf,png,mp4
```

Outputs go to `<run>/figures/`.
Intermediate files go to `<run>/.figure_build/`.
The `confidence` target needs intervals saved by an evaluation with `--bootstrap`.

Use [shared/cases/paper.json](shared/cases/paper.json) when capturing cases
for the main layout. It selects video and audio from the standard manifest:

```sh
python figures/build.py build qualitative-main \
  --run runs/av1m \
  --formats pdf,png
```

Recreate Figure 3 from its saved C22 counts:

```sh
python figures/build.py build calibration \
  --formats pdf,png
```

Render the seven C22 case packages listed in
[shared/cases/index.json](shared/cases/index.json):

```sh
python figures/build.py build gallery \
  --formats pdf,png
```

Standalone builds write to `outputs/figures/` in the repository.
`--output` selects another directory.
Local media can be supplied with `--media-root`.
This directory uses case IDs, with paths such as `audio-only_02/data/source.mp4`.
New runs use the media paths saved by the figure-data exporter.

Add `--calibration-run` or `--comparison-run` to overlay another saved run.
Comparison runs must use the same AP/AR grids.
`--metrics` accepts a saved metrics file directly.
`--cases` selects a local case-package index.

PDFs and videos share their panel layout and coordinates.
Video playback uses the run's score timebase: 25 Hz for constant-rate
localization, or `n_frames / duration` for per-clip timing.
`media_time_origin_s` locates score time zero in the source media.

## Figure data

The evaluation workflow calls:

```python
run_detection(inputs, config, figure_video_ids=("example/clip.mp4",))
# Run the normal final proposal ranking and optional boundary-curve capture.
export_figure_data(run_dir, cases_path, manifest_path, labels_path)
```

`run_detection` saves exact scalar/vector nulls, reference tables, corrections,
and fusion weights under `<run>/fitted/` (or an explicit `fitted_dir`).
It captures selected float64 sequences in `scientific_tracks.npz`, including each
clip's actual fit round. The standard `frame_tracks.npz` format is unchanged.

Scientific display tracks:

- Scalar figure tracks are `(raw - mu) / sigma_w`.
- Synchrony is the mean squared whitened vector component.
- The fused trace uses the fitted frame weights.

A case JSON can be a list, `{"cases": [...]}`, or an object keyed by case ID:

```json
[
  {
    "id": "example",
    "video_id": "example/clip.mp4",
    "crop": [0.0, 1.2],
    "selected_tracks": ["cos", "sync", "fused", "flux"],
    "media": {"video": true, "audio": true},
    "modality_labels": {"video": false, "audio": true}
  }
]
```

| Case field | Meaning |
| --- | --- |
| `id` | Unique directory name. |
| `crop` | Optional, in scored-clip seconds; the default is the complete scored clip. |
| `selected_tracks` | Gallery track names are `cos`, `ja_B`, `jv_B`, `sync`, `vart_C`, `aart_C`, `fused`, and optional `flux`. Scientific group names also work. Absent requested tracks are errors. |
| `media` | Optional media selection. `video` and `audio` can be true or explicit paths relative to the source manifest after expanding environment variables and `~`. Unresolved environment variables are errors. |
| `media.frame_times_s` | Optional; selects nine thumbnail times within the crop. |

The source manifest is a CSV with unique `video_id`, optional
`video_path`/`audio_path`, and optional `media_time_origin_s` (default zero).
Labels use CAST's
`video_id,target,fake_segments_json` schema with half-open seconds.

The exporter reads `proposals_ranked.csv` as the final ranking output.
It writes half-open proposal times from inclusive frame bounds.
It computes maximum per-proposal IoU against GT through CAST's float32 evaluator
function.
IoU inputs use the evaluator's shared `proposal_seconds` conversion; displayed
times retain natural `frame / rate` precision.
It never reorders proposals by an alternative score.

Canonical ranking curves are captured at `boundary_curves/<clip_index>.npz`
with `video_id`, `time_seconds`, and `full_band_flux_percentile`.
The exporter copies them into `boundary_curve.csv`.
These times remain on the canonical 100 Hz untrimmed
waveform clock, including for per-clip localization timebases. Figure code
does not implement boundary flux or change the boundary cue's 25 Hz frame rule.

`figure_data/cases.json` maps source IDs to per-case `data/` directories.
Those contain:

- Full-clip `tracks.csv`, `proposals.csv`, `selected_case.json`, and
  `reference_display_transform.json`.
- Exact `fitted_models.npz` and `fusion_weights.npz`.
- Optional `boundary_curve.csv`.

Media paths and time origins remain explicit; media is decoded only during a
requested render.

## Calibration and gallery

With `--run`, calibration reads the workflow's learned complete-scan
`calibration.csv` with `alpha,rejections,n_genuine,rate`.
`--calibration-run` adds other runs on the same axes.
Without a run, the target reads the paper's counts from
`fig3-statistical-plots/calibration.csv`, grouped by dataset.
Curves use `rejections / n_genuine`; shaded bands are pointwise 95% Wilson intervals.
The paper profile fixes the axis ranges, labels, legend, and crop.
Rates must agree with the counts.

The seven C22 cases in Fig. S4-S10 contain the scientific CSV/JSON data,
PDFs, videos, and playback timing files.
`shared/cases/paper.json` selects the same clip IDs for new runs.
The gallery command renders their scientific panels.
Matching local media and captured ranking curves add media rows and flux.
