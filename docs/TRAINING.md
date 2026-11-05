# Training and fitting

Run these commands from the repository root in the Python 3.12 environment.
Prepare the manifests in [DATA.md](DATA.md) and the features in
[EXTRACTION.md](EXTRACTION.md).

## Protocol fitting

Build the score caches and [corruption corpus](#controlled-corruption-corpus).
Fit the reference models and fusion weights with the supplied heads:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python scripts/train.py \
  --protocol avdeepfake1m_external_reference \
  --out runs/av1m
```

The run saves fitted parameters under `runs/av1m/fitted/`.
Evaluate with the supplied proposal ranker:

```sh
python scripts/evaluate.py \
  --run runs/av1m \
  --out runs/av1m \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --bootstrap 20000
```

| Protocol | Reference data |
| --- | --- |
| External reference | Genuine training clips, split by speaker into fitting, calibration, and fusion pools |
| Four-fold | Reference roles rotate over speaker folds outside the evaluation fold |
| Transfer | Source-dataset references and calibration |

## Neural heads

Defaults are in [heads.json](../configs/heads.json).
Select another file with `--config`.
Explicit command arguments override the file settings.

Extract the primary encoder's clean reference features and the
[artifact-training blends](EXTRACTION.md#artifact-head-features).
Train all seven checkpoints:

```sh
python scripts/train.py heads \
  --features "$CAST_WORK_DIR/features/avdeepfake1m_reference-primary" \
  --visual-blend "$CAST_WORK_DIR/features/artifact-visual" \
  --audio-blend "$CAST_WORK_DIR/features/artifact-audio" \
  --out "$CAST_WORK_DIR/heads" \
  --device cuda
```

The recorded alignment-head pool contains 45,039 genuine training clips:
40,351 for training and 4,688 for validation.
Validation speakers satisfy `SHA1(speaker) % 10 == 0`.
Training samples 160-frame windows from eligible clips.

Artifact heads use the self-splice source clips with matching clean features.

- Validation uses every tenth sorted speaker, starting with the first.
- Cross-fit training first partitions speakers by `SHA1("s5" + speaker) % 2`.
  The validation split is applied within each partition.
- Each corpus clip is scored by the head trained on the opposite partition.

| Checkpoint under `$CAST_WORK_DIR/heads` | Used for |
| --- | --- |
| `prediction_heads.pt` | Alignment/reference heads, synchrony PCA, predictive PCA and Student-t predictors |
| `visual_artifact_head.pt` | Runtime visual-artifact scores |
| `audio_artifact_head.pt` | Runtime audio-artifact scores |
| `visual_artifact_crossfit_{0,1}.pt` | Visual-artifact scores for the corruption corpus |
| `audio_artifact_crossfit_{0,1}.pt` | Audio-artifact scores for the corruption corpus |

The visual heads use dilated convolutions and boundary supervision.
`training.json` records the effective head-training settings.

| Training stage | Optimizer | Default training length |
| --- | --- | --- |
| Alignment | AdamW | 3,000 steps |
| Reference heads | AdamW | 3,000 steps |
| Each Student-t predictor | AdamW | 5,000 steps |
| Each artifact head | Adam | 8 epochs |

- Default learning rate: `3e-4`.
- Seed: 0.
- Alignment/reference/predictor batch size: 48.

Artifact checkpoints retain the lowest validation loss.
`python scripts/train.py heads --help` lists the training options.

For repeated training, build a memory-mapped feature cache:

```sh
python scripts/train.py feature-cache \
  --features "$CAST_WORK_DIR/features/avdeepfake1m_reference-primary" \
  --out "$CAST_WORK_DIR/features/reference"
```

Pass `--features "$CAST_WORK_DIR/features/reference"` to the training command.

## Controlled-corruption corpus

Extract clean, visual-blend, and audio-blend features with both encoders
using the [corpus extraction recipe](EXTRACTION.md#corpus-features).
Keep nine contiguous shards per extraction.
Source order and shard boundaries determine the corruption draws and donors.

Select the supplied checkpoints:

```sh
export HEAD_ROOT="$PWD/models/checkpoints"
```

For retrained heads, set `HEAD_ROOT="$CAST_WORK_DIR/heads"`.
Build the corpus:

```sh
python scripts/prepare.py corpus \
  --primary-clean "$CAST_WORK_DIR/features/self-splice-primary-clean" \
  --auxiliary-clean "$CAST_WORK_DIR/features/self-splice-auxiliary-clean" \
  --primary-visual-blend "$CAST_WORK_DIR/features/self-splice-primary-visual" \
  --auxiliary-visual-blend "$CAST_WORK_DIR/features/self-splice-auxiliary-visual" \
  --primary-audio-blend "$CAST_WORK_DIR/features/self-splice-primary-audio" \
  --auxiliary-audio-blend "$CAST_WORK_DIR/features/self-splice-auxiliary-audio" \
  --source-manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv" \
  --prediction-heads "$HEAD_ROOT/prediction_heads.pt" \
  --visual-crossfit "$HEAD_ROOT/visual_artifact_crossfit_0.pt" "$HEAD_ROOT/visual_artifact_crossfit_1.pt" \
  --audio-crossfit "$HEAD_ROOT/audio_artifact_crossfit_0.pt" "$HEAD_ROOT/audio_artifact_crossfit_1.pt" \
  --output "$CAST_WORK_DIR/models/self_splice_corpus.npz" \
  --device cuda
```

The source manifest selects 3,600 clips from 42 speakers.
The recorded extraction provides 3,599 matched sources.

| Corpus rows | Count |
| --- | ---: |
| Genuine | 3,599 |
| Visual blend | 3,599 |
| Audio blend | 3,599 |
| Audio replacement | 2,119 |
| Audio shift | 1,479 |

The generator uses `mu=2.093`, `sigma=0.592`, and a 33-frame cap.
The adjacent `.build.json` stores source mappings and realized intervals
for audio boundary-cue fitting.
Reference fitting applies `manifests/reference_exclusion.csv` to remove
the source speakers.

For retrained heads, build the
[reference and evaluation score caches](EXTRACTION.md#score-caches)
with the same checkpoints before fitting.

## Proposal-ranker fitting

Fit the reference models, fusion weights, and proposal ranker:

```sh
python scripts/train.py \
  --protocol avdeepfake1m_external_reference \
  --out runs/av1m-refit \
  --fit-ranker \
  --self-splice-manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv"
```

The run saves `proposal_ranker.json`.
Evaluate with `--run runs/av1m-refit --out runs/av1m-refit`.

The ranker uses `log1p_rank`, `adjusted_score`, and `audio_boundary_cue`.
It fits logistic regression with up to 25 Newton-IRLS iterations and ridge
`1e-4` on weights and intercept.
Positive assignments use one-to-one matching at IoU above 0.5.
Fitting excludes rows without measurable selected features.

Audio boundary-cue fitting uses native-gain centered donor excerpts,
hard cuts, and circular five-frame shifts.
Feature-space audio shifts use the corpus generator's sampled offsets.

## Genuine-speech duration fits

Fit duration records from a word-alignment CSV with
`video_id`, `word`, `start_s`, and `end_s`:

```sh
python scripts/prepare.py durations \
  --word-alignments aligned_words.csv \
  --out-dir models
```

Content-word durations determine the lognormal fit.
Genuine interior inter-pausal units determine the 127-frame maximum scan duration.
