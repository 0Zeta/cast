# Reference results

These JSON records contain numerical reference results for the released
method: metrics, populations, registered evaluation grids, and uncertainty.
Values are fractions, rather than percentages. The filename identifies the
dataset and protocol; `ablations/` contains the removal and cumulative panels.

The benchmark recall grid is IoU 0.50:0.05:0.95. The GEM-TFL comparison uses
0.1:0.1:0.7. The AV-Deepfake1M++ complete-validation view keeps all 77,326
metadata rows, while its scored-only view contains 76,874 clips. Compare the
matching protocol, preprocessing, population, and metric view.

The evaluation command writes `metrics.json`, `localization.json`, and `detection.json` for a run.
See [evaluation instructions](../docs/EVALUATION.md) for commands and uncertainty settings.
