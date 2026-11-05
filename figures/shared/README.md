# Shared figure library

Figure folders group paper inputs and layouts by type.
Each type keeps its components and styles with its layout.
This directory holds common styles, assets, fonts, case indexes, and Python helpers.
See [the figure guide](../README.md) for the folder map and build commands.

| Path | Role |
| --- | --- |
| `styles/illustration.typ` | Common illustration styles |
| `styles/tokens.toml` | Shared colors, fonts, line widths, and spacing |
| `assets/frames/` | Original paper frames 30, 32, ..., 52 |
| `assets/frames_strip_vertical.png` | Method figure's twelve-frame strip |
| `assets/audio_waveform.tex` | Measured 144-bar waveform from 1.20 to 2.16 s |
| `assets/illustration_inputs.json` | Paper clip ID, frame selections, and waveform interval |
| `assets/mathcal_I_x.svg` | Interval-set glyph |
| `fonts/` | Fonts and their licenses |

`prepare_illustrations.py` rebuilds the strip from the original paper frames.
The measured waveform is stored alongside them.
The Python helpers prepare media and display coordinates, compile Typst,
combine PDFs, and encode video from saved scientific inputs.
Statistical profiles in
[profiles.json](../fig3-statistical-plots/profiles.json) use
[the statistical layout](../fig3-statistical-plots/figure.typ).
The main and supplementary qualitative views use
[the qualitative layout](../fig4-qualitative-examples/figure.typ).
The builder loops over profiles or cases through the same renderer for each type.
PDF and video panels share the exported layout geometry.

| Case file | Role |
| --- | --- |
| `cases/paper.json` | Paper clip IDs and manifest video/audio selections for new runs |
| `cases/example.json` | Example selection to adapt to an evaluation manifest |
| `cases/index.json` | Stored C22 case IDs, paper figure labels, and data directories |

Stored cases live under `fig4-qualitative-examples/cases/<logical-case-id>/`.
The stored-case index follows Fig. S4-S10 order.
Each entry has `id`, `paper_figure`, and a `data` path relative to the index.
Case selections and local media directories use the logical case IDs.

Run-based builds write to `<run>/figures/`.
Standalone builds default to `outputs/figures/` at the repository root.
