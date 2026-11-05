# Data and manifests

Prepare manifests from the official metadata and local benchmark directories.
A manifest has `video_id`, `video_path`, and `target` (0 genuine, 1 forged).
Localization also needs `fake_segments_json`, a list of `[start, end]` pairs
in seconds; genuine rows use `[]`. Relative media paths resolve from the
manifest directory, and environment variables in paths are expanded.

Keep the generated `video_id` values: they determine joins and speaker folds.
`video_path` locates media and can point to a different disk. Recorded
`pool_split`, `calibration_seed`, and `fusion_seed` columns preserve the
reference experiment's genuine-pool sampling independently of disk paths.
They are computational inputs; use execution seed 0 with these columns.

## AV-Deepfake1M and AV-Deepfake1M++

```sh
export AVDEEPFAKE1M_ROOT=/path/to/AV-Deepfake1M
export AVDEEPFAKE1MPP_ROOT=/path/to/AV-Deepfake1M-PlusPlus
export CAST_WORK_DIR=/path/to/cast-work

python scripts/prepare.py evaluation --dataset avdeepfake1m \
  --metadata "$AVDEEPFAKE1M_ROOT/val_metadata.json" \
  --dataset-root "$AVDEEPFAKE1M_ROOT" \
  --output "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv"
python scripts/prepare.py evaluation --dataset avdeepfake1m-plus \
  --metadata "$AVDEEPFAKE1MPP_ROOT/val_metadata.json" \
  --dataset-root "$AVDEEPFAKE1MPP_ROOT" \
  --output "$CAST_WORK_DIR/manifests/avdeepfake1m_plus_validation.csv"
```

The audiovisual manifests contain 57,340 and 76,928 rows. For complete
AV-Deepfake1M++ evaluation, repeat its command with `--include-silent-videos`
and output `manifests/avdeepfake1m_plus_complete.csv`; this retains all
77,326 rows, including 398 silent-video records.

For the registered comparison lists, repeat the AV-Deepfake1M command with
`--cohort avh-align-avpf-10000` or `--cohort boldisor-5133`, writing
`avdeepfake1m_shared_10000.csv` or `avdeepfake1m_boldisor_5133.csv` under
`manifests/`. Their bundled cohort indices preserve membership and order.

## Genuine references and controlled-corruption sources

```sh
python scripts/prepare.py reference \
  --train-metadata "$AVDEEPFAKE1M_ROOT/train_metadata.json" \
  --exclude-validation-manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --exclude-validation-manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_plus_validation.csv" \
  --output "$CAST_WORK_DIR/manifests/avdeepfake1m_reference.csv"
python scripts/prepare.py sources \
  --genuine-manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_reference.csv" \
  --train-metadata "$AVDEEPFAKE1M_ROOT/train_metadata.json" \
  --sources-out "$CAST_WORK_DIR/manifests/self_splice_sources.csv" \
  --reference-exclusion-out "$CAST_WORK_DIR/manifests/reference_exclusion.csv"
```

| Input | Composition |
| --- | --- |
| Ordered reference manifest | 45,063 rows |
| Selected sources | 1,200 rows from each contiguous third, spanning 42 speakers |
| Joined sources | 3,599; the availability column records the one excluded source |

The reference-exclusion list removes these source speakers before fitting.
Preserve source order and speaker-bearing identifiers.

## Transfer dataset

```sh
python scripts/prepare.py transfer lavdf --root /path/to/LAV-DF \
  --metadata /path/to/LAV-DF/metadata.min.json \
  --output "$CAST_WORK_DIR/manifests/lav_df.csv"
```

Export `LAVDF_ROOT` to the LAV-DF media root before extraction. LAV-DF
manifests retain metadata duration for per-clip localization timing.

## Score caches

Each `{set}_{main,avsr,artifact}.npz` contains `video_ids`, `lengths`,
`offsets`, `scores`, and `score_names`. Offsets start at zero and end at the
number of score rows. The synchrony cache uses the same identifiers and
offsets with one array per vector score. Required groups and cache paths
are specified by the protocol configuration.

Joined caches may differ by at most two frames and are truncated to their
common length. Reference scoring preserves the registered attrition:

| Population | Scored / manifest clips |
| --- | --- |
| AV-Deepfake1M | 57,315/57,340 |
| AV-Deepfake1M++ | 76,874/76,928 |
| Shared list | 9,997/10,000 |
| Boldisor | 5,132/5,133 |

Use `missing_rows: "raise"` for other populations.
The registered LAV-DF cache has no scoring losses.
