# CAST

Code for **Audio-Visual Temporal Forgery Localization without Training on Deepfakes**

Lucas Schnack, Florian Hunecke, Mariana-Iuliana Georgescu, Radu Tudor Ionescu, A. Sophia Koepke

WACV 2027

CAST localizes temporal forgeries using frozen AV-HuBERT features and genuine
reference clips. It combines five scalar score sequences and a vector synchrony
residual through 22 interval-test channels. A cross-clip proposal ranker learns
from controlled corruptions.

## Install

Use Python 3.12. Install `ffmpeg` and `ffprobe` for audio decoding.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[cache]'

export CAST_WORK_DIR=/path/to/cast-work
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Encoder extraction uses the separate Python 3.10 environment in
[EXTRACTION.md](docs/EXTRACTION.md).

## Run with supplied heads

The example uses the AV-Deepfake1M external-reference protocol.
Neural checkpoints are in `models/checkpoints/`.
Obtain benchmark media and metadata from their distributors.

Prepare the inputs under `CAST_WORK_DIR` in this order:

| Step | Command | Input -> output | Guide |
| --- | --- | --- | --- |
| 1. Prepare data | `scripts/prepare.py evaluation`, `reference`, `sources` | Benchmark metadata and media paths -> manifests | [Data](docs/DATA.md) |
| 2. Download assets | `scripts/prepare.py assets` | Upstream downloads -> encoders and mouth-alignment assets | [Extraction](docs/EXTRACTION.md#environments-and-assets) |
| 3. Extract features | `scripts/extract.py features` | Media and encoders -> feature TARs | [Extraction](docs/EXTRACTION.md#encoder-features) |
| 4. Build scores | `scripts/extract.py scores` | Features and supplied heads -> evaluation/reference score caches | [Extraction](docs/EXTRACTION.md#score-caches) |
| 5. Build corpus | `scripts/prepare.py corpus` | Six source-feature extractions and cross-fit heads -> corruption corpus | [Training](docs/TRAINING.md#controlled-corruption-corpus) |

Extract evaluation and reference features with both encoders. Corpus sources
need clean, visual-blend, and audio-blend features from each encoder.
Use nine contiguous shards per corpus extraction.

Fit the genuine-reference models and fusion weights:

```sh
python scripts/train.py \
  --protocol avdeepfake1m_external_reference \
  --out runs/av1m
```

Evaluate using the supplied proposal ranker:

```sh
python scripts/evaluate.py \
  --run runs/av1m \
  --out runs/av1m \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --bootstrap 20000
```

| Stage | Output |
| --- | --- |
| Fit | `runs/av1m/fitted/`, clip scores, proposals |
| Evaluate | `runs/av1m/proposals_ranked.csv`, `metrics.json`, `calibration.csv` |

Re-evaluating `--run` reuses the saved scan.
[Protocol files](configs/protocols/) define input paths and evaluation settings.
Each protocol identifies its citations, population, preprocessing, and fitting.
[EVALUATION.md](docs/EVALUATION.md) covers four-fold fits, transfer, and comparison cohorts.

## Pipeline

The pipeline command prepares inputs, extracts features, builds scores and
the corruption corpus, fits the method, and evaluates it.
Set the dataset roots described in [DATA.md](docs/DATA.md).

```sh
python scripts/reproduce.py \
  --protocol avdeepfake1m_external_reference \
  --out runs/av1m-pipeline \
  --extract-python .venv-avhubert/bin/python \
  --avhubert-root /path/to/av_hubert \
  --bootstrap 20000
```

Use `--from fit` to start with prepared score caches and a corruption corpus.
Add `--train-heads` to train the heads and refit the proposal ranker.
Add `--figure-cases figures/shared/cases/example.json` to render the selected cases
from the same run.

## Figures

Install the [figure renderer](figures/README.md).
Set the clip IDs in [example.json](figures/shared/cases/example.json)
to match the evaluation manifest.
When creating the run above, add `--figure-cases figures/shared/cases/example.json`
to both the fitting and evaluation commands.

```sh
python figures/build.py build all \
  --run runs/av1m \
  --formats pdf,png,mp4
```

Outputs go to `runs/av1m/figures/`.
Figures read the run's saved scores, proposals, boundary curves, and calibration.
Standalone builds write to `outputs/figures/` in the repository.
Folders under `figures/` group the paper inputs and stored figures by type.
Each type folder holds its layout, components, and styles.
`figures/shared/` holds common styles, assets, fonts, case lists, and Python helpers.
[Figure options](figures/README.md) describe case selection and render targets.

## Retrain CAST

This route adds head training before steps 4-5.
Extract genuine source clips with the primary encoder and
`--blend-purpose artifact-head`.
Use `--mode visual-blend` for `$CAST_WORK_DIR/features/artifact-visual`.
Use `--mode audio-blend` for `$CAST_WORK_DIR/features/artifact-audio`.

Defaults are in [heads.json](configs/heads.json).
Train the neural heads on the primary encoder's genuine reference features:

```sh
python scripts/train.py heads \
  --features "$CAST_WORK_DIR/features/avdeepfake1m_reference-primary" \
  --visual-blend "$CAST_WORK_DIR/features/artifact-visual" \
  --audio-blend "$CAST_WORK_DIR/features/artifact-audio" \
  --out "$CAST_WORK_DIR/heads" \
  --device cuda
```

The command writes seven checkpoints under `$CAST_WORK_DIR/heads`.
Build the scores and corpus using those checkpoint paths:

| Command | Checkpoint arguments |
| --- | --- |
| `scripts/extract.py scores` | `--prediction-heads`, `--visual-artifact-head`, `--audio-artifact-head` |
| `scripts/prepare.py corpus` | `--prediction-heads`, `--visual-crossfit` (two paths), `--audio-crossfit` (two paths) |

Fit the reference models, fusion weights, and proposal ranker:

```sh
python scripts/train.py \
  --protocol avdeepfake1m_external_reference \
  --out runs/av1m-retrained \
  --fit-ranker \
  --self-splice-manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv"
```

Evaluate with `--run runs/av1m-retrained --out runs/av1m-retrained`.
[TRAINING.md](docs/TRAINING.md) specifies corpus construction and fitting settings.

## Ablations

```sh
python scripts/ablations.py \
  --protocol avdeepfake1m_external_reference \
  --arm without_audio_boundary_cue \
  --run runs/av1m \
  --out runs/no-boundary-cue \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --self-splice-manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv"
```

[Ablation definitions](configs/ablations/definitions.json) specify each arm's
changes and fitting policy.

## Reference results

AR uses the benchmark IoU grid, 0.50:0.05:0.95.
GEM-TFL recall uses 0.1:0.1:0.7.
[EVALUATION.md](docs/EVALUATION.md) defines both views.
Values are fractions.

| Dataset | Protocol | AP@0.5 | AP@0.75 | AR@50 | AUC |
|---|---|---:|---:|---:|---:|
| AV-Deepfake1M | External reference, 57,315 scored clips | 0.422 | 0.139 | 0.495 | 0.919 |
| AV-Deepfake1M | Four-fold, 57,315 scored clips | 0.427 | 0.132 | 0.484 | 0.918 |
| AV-Deepfake1M++ | Four-fold, 76,874 scored clips | 0.316 | 0.087 | 0.462 | 0.798 |
| LAV-DF | Transfer, 26,100 clips | 0.413 | 0.151 | 0.528 | 0.965 |

Full results are under [results/](results/).

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{schnack2027cast,
  title     = {Audio-Visual Temporal Forgery Localization without Training on Deepfakes},
  author    = {Schnack, Lucas and Hunecke, Florian and Georgescu, Mariana-Iuliana and Ionescu, Radu Tudor and Koepke, A. Sophia},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License and acknowledgments

CAST code uses the [MIT license](LICENSE).
AV-HuBERT, Fairseq, dlib, the mean-face template from
`mpc001/Lipreading_using_Temporal_Convolutional_Networks`, and each benchmark
retain their respective terms.
Evaluation follows the AV-Deepfake1M reference evaluator.
The teaser and overview use illustrative media.
