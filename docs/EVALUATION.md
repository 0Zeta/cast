# Evaluation

Each named configuration under `configs/protocols/` contains input paths,
method settings, a proposal ranker, and the evaluation population and rules.
`--store` or `CAST_WORK_DIR` resolves dataset-local paths. The experiment's
output directory retains these resolved settings and the fitted parameters.

| Field | Contents |
| --- | --- |
| `references` | Method, dataset, comparison, and evaluator citations |
| `population` | Dataset split, fixed cohort, and sampling identities |
| `preprocessing` | Encoder checkpoints, audio rate, frame cap, and trimming |
| `fitting` | Reference source, speaker partition, and ranker training data |
| `evaluation` | Detection settings, localization views, and bootstrap rules |

Shared method settings are in [method.json](../configs/method.json).
`inputs`, `ranking`, and `evaluation` specify the computation.
`preprocessing` describes the extraction used for the input caches.
The loader checks metric names and localization views against the evaluator.

```sh
python scripts/evaluate.py \
  --protocol avdeepfake1m_external_reference \
  --out runs/av1m \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --bootstrap 20000
```

The command fits the declared genuine-reference model, scans, computes audio
boundary cues, ranks proposals, and writes localization and detection metrics.
For an existing scan or fitted run:

```sh
python scripts/evaluate.py \
  --run runs/av1m \
  --out runs/av1m \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --bootstrap 20000
```

The second command reuses fitted parameters and scan emissions. Point
estimates alone are obtained by omitting `--bootstrap`. The ordinary output
files include `clip_scores.csv`, `proposals_ranked.csv`, `detection.json`,
`localization.json`, `metrics.json`, and `calibration.csv`.

## Localization views

Both views use the reference evaluator's matching and float32 arithmetic,
including strict IoU comparison and stable tie ordering. Frame bounds are
inclusive: `[start_frame, end_frame]` maps to
`[start_frame/rate, (end_frame+1)/rate)` seconds.

| View | AP IoU thresholds | Recall IoU grid | Proposal budgets |
|---|---|---|---|
| AV-Deepfake1M benchmark | 0.5, 0.75, 0.9, 0.95 | 0.50:0.05:0.95 | 50, 30, 20, 10, 5 |
| GEM-TFL comparison | 0.1:0.1:0.7 | 0.1:0.1:0.7 | 20, 10, 5, 2 |

The views are evaluated separately. Recall computed on one grid is not
substituted for recall on the other. These settings follow the
[benchmark evaluator](https://github.com/ControlNet/AV-Deepfake1M#sdk) and
[GEM-TFL](https://arxiv.org/pdf/2603.05095).

AV-Deepfake1M uses 25 fps. LAV-DF uses each clip's emitted frame count divided
by its metadata duration. Protocol configuration selects the correct rule.
The optional compiled-evaluator comparison uses an absolute tolerance of
`1e-6`; `avdeepfake1m==0.0.4` can be installed with `pip install --no-deps`.

The four-fold AV-Deepfake1M++ configuration evaluates scored clips and also
provides the complete-validation view when given the complete 77,326-row
manifest. The latter assigns empty predictions to all 452 unscored rows:

- 398 have no audio stream.
- 54 are extraction losses.

These populations remain distinct.
The complete-validation view reports point estimates.

## Detection and uncertainty

Detection reports ROC-AUC, average precision, empirical EER, TPR at fixed
FPR, and FPR at fixed TPR.

- **Primary statistic:** `-log(clip_pvalue)`.
- **Realized decisions at nominal levels:** `exact_clip_rank`.
- **Secondary p-value panels:** retain their own finite populations.

A registered cohort is selected before evaluation. Full-run predictions can
therefore serve the smaller shared 10,000-clip or Boldisor lists without
changing the reference/calibration fit. Trimmed protocols extract both
modalities after leading-silence removal; trimming is not replayed on cached
features.

Confidence intervals use 20,000 draws when `--bootstrap 20000` is supplied.
Localization resamples speakers, or clips for LAV-DF.
Detection resampling follows each configuration:

| Protocols | Resampling | Seed | Batch size |
| --- | --- | ---: | ---: |
| Main protocols | Multinomial cluster multiplicities | 1 | 128 |
| Matched cohorts | Index resampling | 0 | 64 |

Percentile intervals use NumPy's linear 2.5th/97.5th percentiles.
SD uses `ddof=0`.
Exact-rank rates use Wilson intervals.
The reference records specify which uncertainty panels accompany each published metric.

`calibration.csv` evaluates genuine clips using `exact_learned_scan_rank`,
the learned scan maximum's exact rank used by Figure 3.

## Ablations

```sh
python scripts/ablations.py \
  --protocol avdeepfake1m_external_reference \
  --arm without_cross_modal_agreement \
  --out runs/no-agreement \
  --manifest "$CAST_WORK_DIR/manifests/avdeepfake1m_validation.csv" \
  --self-splice-manifest "$CAST_WORK_DIR/manifests/self_splice_sources.csv" \
  --bootstrap 20000
```

- **Scan-component arms:** refit their detection objects and proposal ranker.
- **Ranking-only arms:** reuse a baseline emission supplied through `--run`.
  Apply the declared ranking change and refit the ranker where required.
- **Cumulative arms:** add score groups in the fixed order of
  `configs/ablations/definitions.json`.

Ablation results use the same metric views and populations as the chosen baseline protocol.
