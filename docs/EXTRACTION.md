# Feature and score extraction

Prepare the media manifests in [DATA.md](DATA.md).
Run the commands below from the CAST repository root.
For custom media, use [extraction-inputs.csv](../examples/extraction-inputs.csv)
as the manifest template.

## Environments and assets

Use Python 3.10 for AV-HuBERT extraction:

```sh
git clone https://github.com/facebookresearch/av_hubert.git /path/to/av_hubert
git -C /path/to/av_hubert checkout 258fb50e155134eec2c4b49c2ae8de267075fd18
git -C /path/to/av_hubert submodule update --init --recursive

python3.10 -m venv .venv-avhubert
. .venv-avhubert/bin/activate
python -m pip install pip==24.0
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/extraction.txt
python -m pip install \
  --no-deps opencv-python==4.13.0.92
python -m pip install -e /path/to/av_hubert/fairseq
```

Choose the PyTorch index for your accelerator.
Keep NumPy 1.23.5 for the extraction environment.
The reference stack used ffmpeg/ffprobe 6.1.1.

Download assets in the Python 3.12 CAST environment:

```sh
python scripts/prepare.py assets \
  --directory /path/to/assets
```

[assets.json](../configs/assets.json) lists the files and upstream licenses.

## Encoder features

Activate the Python 3.10 extraction environment:

```sh
. .venv-avhubert/bin/activate

export CAST_ROOT=/path/to/cast
export AVH_ROOT=/path/to/av_hubert
export ASSET_ROOT=/path/to/assets
export CAST_WORK_DIR=/path/to/cast-work
export PYTHONPATH="$CAST_ROOT/src"

python "$CAST_ROOT/scripts/extract.py" features \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --output-dir "$CAST_WORK_DIR/features/avdeepfake1m_validation-primary" \
  --avhubert-root "$AVH_ROOT" \
  --checkpoint "$ASSET_ROOT/large_vox_iter5.pt" \
  --cnn-detector "$ASSET_ROOT/mmod_human_face_detector.dat" \
  --face-predictor "$ASSET_ROOT/shape_predictor_68_face_landmarks.dat" \
  --mean-face "$ASSET_ROOT/20words_mean_face.npy" \
  --mode clean \
  --rank 0 \
  --nshard 1 \
  --device cuda
```

Repeat for both encoders:

| Encoder | Checkpoint | Output suffix |
| --- | --- | --- |
| Primary | `large_vox_iter5.pt` | `-primary` |
| Auxiliary | `self_large_vox_433h.pt` | `-auxiliary` |

Repeat for `avdeepfake1m_reference.csv` with output prefix
`avdeepfake1m_reference`.
Use `lav_df.csv` with output prefix `lav_df` for the LAV-DF transfer protocol.

Each TAR record contains audio, visual, and joint features.
Preprocessing aligns 96-by-96 mouth crops and stacks mono 16-kHz filterbanks
at 25 Hz.
Clips are capped at 750 frames.
`summary_shard_*.csv` records extraction status, coverage, and truncation.
Use a fresh output directory for each extraction recipe.

For the shared 10k AVH-Align/AVPF protocol, add `--leading-silence-trim` and
`--trim-order cap_then_trim` to both encoder commands.
Write the outputs to the corresponding `_trimmed` directories.
The pipeline command reads the trim order from the protocol's `preprocessing` field.
Trimmed extraction records `media_time_origin_s` in its summaries.
Reference and corruption-source extractions use untrimmed media.

## Score caches

Activate the Python 3.12 CAST environment.
Build validation scores with the supplied heads:

```sh
. .venv/bin/activate
python -m pip install -e '.[cache]'

export HEAD_ROOT="$CAST_ROOT/models/checkpoints"

python scripts/extract.py scores \
  --primary-features "$CAST_WORK_DIR/features/avdeepfake1m_validation-primary" \
  --auxiliary-features "$CAST_WORK_DIR/features/avdeepfake1m_validation-auxiliary" \
  --prediction-heads "$HEAD_ROOT/prediction_heads.pt" \
  --visual-artifact-head "$HEAD_ROOT/visual_artifact_head.pt" \
  --audio-artifact-head "$HEAD_ROOT/audio_artifact_head.pt" \
  --output-dir "$CAST_WORK_DIR/scores" \
  --prefix avdeepfake1m_validation \
  --device cuda
```

For retrained heads, change `HEAD_ROOT` to `$CAST_WORK_DIR/heads`.
Use an empty score directory for each head set.
Score references and evaluation clips with the same heads.

Repeat for `avdeepfake1m_reference` and each target dataset.
The command writes
`{prefix}_{main,avsr,sync,artifact}.npz` under `$CAST_WORK_DIR/scores`.
The [protocol files](../configs/protocols/) specify the expected prefixes.
[DATA.md](DATA.md#score-caches) describes the cache format.

## Corpus features

Prepare `self_splice_sources.csv` as in [DATA.md](DATA.md).
In the Python 3.10 environment, extract the primary clean features:

```sh
. .venv-avhubert/bin/activate

python "$CAST_ROOT/scripts/extract.py" features \
  --manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv" \
  --output-dir "$CAST_WORK_DIR/features/self-splice-primary-clean" \
  --avhubert-root "$AVH_ROOT" \
  --checkpoint "$ASSET_ROOT/large_vox_iter5.pt" \
  --cnn-detector "$ASSET_ROOT/mmod_human_face_detector.dat" \
  --face-predictor "$ASSET_ROOT/shape_predictor_68_face_landmarks.dat" \
  --mean-face "$ASSET_ROOT/20words_mean_face.npy" \
  --mode clean \
  --blend-purpose corpus \
  --rank 0 \
  --nshard 9 \
  --device cuda
```

- Run each extraction for ranks 0 through 8.
- Keep the source manifest order.

Use these modes and output directories for the six extractions:

| Encoder | Mode | Directory under `$CAST_WORK_DIR/features` |
| --- | --- | --- |
| Primary | `clean` | `self-splice-primary-clean` |
| Auxiliary | `clean` | `self-splice-auxiliary-clean` |
| Primary | `visual-blend` | `self-splice-primary-visual` |
| Auxiliary | `visual-blend` | `self-splice-auxiliary-visual` |
| Primary | `audio-blend` | `self-splice-primary-audio` |
| Auxiliary | `audio-blend` | `self-splice-auxiliary-audio` |

Select `self_large_vox_433h.pt` for each auxiliary extraction.
The corpus blend law uses `mu=2.093`, `sigma=0.592`, and cap 33 frames.
Shard boundaries determine random draws and donor chains.
[Build the corpus](TRAINING.md#controlled-corruption-corpus) in Python 3.12.

## Artifact-head features

Use the primary encoder to extract self-blends for
[neural-head training](TRAINING.md#neural-heads):

```sh
. .venv-avhubert/bin/activate

python "$CAST_ROOT/scripts/extract.py" features \
  --manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv" \
  --output-dir "$CAST_WORK_DIR/features/artifact-visual" \
  --avhubert-root "$AVH_ROOT" \
  --checkpoint "$ASSET_ROOT/large_vox_iter5.pt" \
  --cnn-detector "$ASSET_ROOT/mmod_human_face_detector.dat" \
  --face-predictor "$ASSET_ROOT/shape_predictor_68_face_landmarks.dat" \
  --mean-face "$ASSET_ROOT/20words_mean_face.npy" \
  --mode visual-blend \
  --blend-purpose artifact-head \
  --rank 0 \
  --nshard 9 \
  --device cuda
```

- Run ranks 0 through 8.
- Repeat with `--mode audio-blend` and output directory
  `$CAST_WORK_DIR/features/artifact-audio`.

The artifact-training blend law uses `mu=1.95`, `sigma=0.5`,
and cap 20 frames.
The clean counterparts come from `avdeepfake1m_reference-primary`.
