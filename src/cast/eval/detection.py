"""Evaluate clip-level detection and bootstrap its uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import speaker_cluster_key

SCORE_FLOOR = 1e-300
RANK_ALPHAS = (0.01, 0.05, 0.10)
TPR_TARGETS = (0.90, 0.95)
RESAMPLING_SCHEMES = ("multinomial", "index")


def detection_score(pvalue) -> np.ndarray:
    """The ranking statistic of a calibrated p-value column: -log p."""
    p = np.asarray(pvalue, dtype=np.float64)
    return -np.log(np.maximum(p, SCORE_FLOOR))


class ScoreTable:
    """Labels and scores sorted once, so weighted metrics cost one pass."""

    def __init__(self, y, score):
        y = np.asarray(y, dtype=np.int8)
        score = np.asarray(score, dtype=np.float64)
        if y.shape != score.shape or y.ndim != 1:
            raise ValueError("labels and scores must be one aligned vector")
        if not np.isfinite(score).all():
            raise ValueError("detection scores must be finite")
        if set(np.unique(y).tolist()) - {0, 1}:
            raise ValueError("labels must be 0 or 1")
        self.n = len(y)
        self.order = np.argsort(score, kind="stable")      # ascending
        s = score[self.order]
        self.y = y[self.order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        self.thresholds = s[self.starts]

    def _runs(self, weights) -> tuple[np.ndarray, np.ndarray]:
        """Weighted negative and positive mass per distinct score, ascending."""
        w = np.atleast_2d(np.asarray(weights, dtype=np.float64))[:, self.order]
        real = np.add.reduceat(w * (self.y == 0), self.starts, axis=1)
        fake = np.add.reduceat(w * (self.y == 1), self.starts, axis=1)
        return real, fake

    def roc_auc(self, weights=None) -> np.ndarray:
        """Tie-corrected ROC-AUC for one weight row or a (draws, n) matrix."""
        real, fake = self._runs(np.ones(self.n) if weights is None else weights)
        real_below = np.cumsum(real, axis=1) - real
        numerator = np.sum(fake * (real_below + 0.5 * real), axis=1)
        denominator = real.sum(axis=1) * fake.sum(axis=1)
        return np.divide(numerator, denominator,
                         out=np.full_like(numerator, np.nan),
                         where=denominator > 0)

    def average_precision(self, weights=None) -> np.ndarray:
        """All-point average precision over distinct score thresholds."""
        real, fake = self._runs(np.ones(self.n) if weights is None else weights)
        tp = np.cumsum(fake[:, ::-1], axis=1)
        fp = np.cumsum(real[:, ::-1], axis=1)
        n_pos, n_neg = tp[:, -1], fp[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = tp / (tp + fp)
            recall = tp / n_pos[:, None]
        gain = np.diff(np.c_[np.zeros(len(tp)), recall], axis=1)
        ap = np.nansum(gain * precision, axis=1)
        return np.where((n_pos > 0) & (n_neg > 0), ap, np.nan)

    def roc_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(fpr, tpr, thresholds) at every distinct score, descending,
        preceded by the empty operating point at threshold +inf."""
        real, fake = self._runs(np.ones(self.n))
        tp = np.cumsum(fake[0, ::-1])
        fp = np.cumsum(real[0, ::-1])
        return (np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]],
                np.r_[np.inf, self.thresholds[::-1]])


def operating_points(table: ScoreTable) -> dict:
    """Equal error rate and fixed-rate operating points of the ROC curve."""
    fpr, tpr, thresholds = table.roc_curve()
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    threshold = float(thresholds[i])
    tpr_at_fpr = {}
    for alpha in RANK_ALPHAS:
        eligible = tpr[fpr <= alpha]
        tpr_at_fpr[f"{alpha:g}"] = float(eligible.max()) if len(eligible) else 0.0
    fpr_at_tpr = {}
    for target in TPR_TARGETS:
        eligible = fpr[tpr >= target]
        fpr_at_tpr[f"{target:g}"] = (float(eligible.min()) if len(eligible)
                                     else float("nan"))
    return {
        "average_precision": float(table.average_precision()[0]),
        "eer": float(0.5 * (fpr[i] + fnr[i])),
        "eer_fpr": float(fpr[i]), "eer_fnr": float(fnr[i]),
        "eer_score_threshold": threshold,
        "eer_pvalue_threshold": float(np.exp(-threshold)),
        "tpr_at_fpr": tpr_at_fpr, "fpr_at_tpr": fpr_at_tpr,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((center - half) / denominator),
            float((center + half) / denominator)]


def exact_rank_rates(y, pvalue, alphas=RANK_ALPHAS) -> dict:
    """Realized genuine false-alarm and forged hit rates of p <= alpha."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(pvalue, dtype=np.float64)
    finite = np.isfinite(p)
    out = {}
    for alpha in alphas:
        row = {}
        for label, rate, population in ((0, "fpr", "real"), (1, "tpr", "fake")):
            mask = finite & (y == label)
            n = int(mask.sum())
            k = int((p[mask] <= alpha).sum())
            row[rate] = {"point": k / n if n else float("nan"),
                         "wilson95": wilson_interval(k, n), "k": k, "n": n,
                         "population": population}
        out[f"{alpha:g}"] = row
    return out


def score_panel(y, pvalue, column: str, *, require_complete: bool) -> dict:
    """Threshold-free and operating-point metrics of one p-value column."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(pvalue, dtype=np.float64)
    finite = np.isfinite(p)
    n_finite = int(finite.sum())
    if require_complete and n_finite != len(p):
        raise ValueError(f"primary score {column} has {len(p) - n_finite} "
                         "non-finite rows")
    if n_finite and ((p[finite] < 0).any() or (p[finite] > 1 + 1e-12).any()):
        raise ValueError(f"{column} leaves the p-value range [0, 1]")
    yf = y[finite]
    population = {"n_total": len(y), "n_finite": n_finite,
                  "n_missing": len(y) - n_finite,
                  "n_real": int((yf == 0).sum()), "n_fake": int((yf == 1).sum())}
    panel = {"column": column, "statistic": f"-log({column})",
             "population": population}
    if n_finite == 0 or len(np.unique(yf)) != 2:
        return {**panel, "defined": False,
                "reason": "finite population does not contain both classes"}
    table = ScoreTable(yf, detection_score(p[finite]))
    return {**panel, "defined": True, "roc_auc": float(table.roc_auc()[0]),
            "operating_points": operating_points(table)}


def cluster_weights(n_clusters: int, size: int, rng: np.random.Generator,
                    scheme: str) -> np.ndarray:
    """(size, n_clusters) resampled cluster multiplicities."""
    if scheme == "multinomial":
        return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters),
                               size=size).astype(np.float64)
    if scheme == "index":
        draws = rng.integers(0, n_clusters, size=(size, n_clusters))
        weights = np.zeros((size, n_clusters))
        np.add.at(weights, (np.repeat(np.arange(size), n_clusters),
                            draws.ravel()), 1.0)
        return weights
    raise ValueError(f"unknown resampling scheme {scheme!r}; "
                     f"expected one of {RESAMPLING_SCHEMES}")


@dataclass(frozen=True)
class DetectionBootstrap:
    n_clusters: int
    n_resamples: int
    seed: int
    scheme: str
    batch_size: int
    #: panel -> metric -> (n_resamples,) draws; NaN marks a class-degenerate draw
    draws: dict[str, dict[str, np.ndarray]]


def interval(draws: np.ndarray) -> dict:
    """Percentile interval and SD over the finite draws, with draw accounting."""
    finite = draws[np.isfinite(draws)]
    if len(finite):
        lo, hi = np.percentile(finite, [2.5, 97.5])
        ci95, sd = [float(lo), float(hi)], float(finite.std())
    else:
        ci95, sd = [float("nan"), float("nan")], float("nan")
    return {"ci95": ci95, "sd": sd, "n_registered": len(draws),
            "n_effective": len(finite), "n_degenerate": len(draws) - len(finite)}


def joint_bootstrap(y, score, cluster_keys, panels: dict[str, np.ndarray], *,
                    n_resamples: int, seed: int, scheme: str = "multinomial",
                    batch_size: int = 128) -> DetectionBootstrap:
    """One cluster-resampling stream applied to every panel mask.

    Each draw resamples clusters with replacement; a clip's multiplicity is
    its cluster's count. ROC-AUC and average precision are recomputed from
    the weighted populations, so paired differences between panels share
    the draw."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    keys = np.asarray([str(k) for k in cluster_keys])
    unique, inverse = np.unique(keys, return_inverse=True)
    tables = {name: ScoreTable(np.asarray(y)[mask], np.asarray(score)[mask])
              for name, mask in panels.items()}
    draws = {name: {"roc_auc": np.empty(n_resamples),
                    "average_precision": np.empty(n_resamples)}
             for name in panels}
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, batch_size):
        size = min(batch_size, n_resamples - start)
        weights = cluster_weights(len(unique), size, rng, scheme)[:, inverse]
        for name, mask in panels.items():
            table = tables[name]
            draws[name]["roc_auc"][start:start + size] = \
                table.roc_auc(weights[:, mask])
            draws[name]["average_precision"][start:start + size] = \
                table.average_precision(weights[:, mask])
    return DetectionBootstrap(n_clusters=len(unique), n_resamples=n_resamples,
                              seed=seed, scheme=scheme, batch_size=batch_size,
                              draws=draws)


def panel_masks(labels, strata: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The overall mask plus one mask per stratum value.

    For ``modify_type`` each forged type is paired with every genuine clip;
    other columns select their own rows, which must hold both classes."""
    target = labels["target"].to_numpy()
    masks = {"overall": np.ones(len(labels), dtype=bool)}
    for column in strata:
        values = labels[column].astype(str).to_numpy()
        if column == "modify_type":
            for value in sorted(set(values[target == 1])):
                masks[f"{column}:{value}"] = (target == 0) | (values == value)
        else:
            for value in sorted(set(values)):
                masks[f"{column}:{value}"] = values == value
    return masks


def evaluate_detection(labels, scores, score_columns: tuple[str, ...], *,
                       primary: str, exact_rank: str,
                       strata: tuple[str, ...] = ("modify_type",),
                       n_resamples: int, seed: int, scheme: str = "multinomial",
                       batch_size: int = 128,
                       cluster_key=speaker_cluster_key) -> dict:
    """Evaluate a clip-score table against its label table.

    ``labels`` and ``scores`` are aligned pandas frames over the scored clips;
    attrition is the caller's concern."""
    if list(labels["video_id"]) != list(scores["video_id"]):
        raise ValueError("labels and scores are not aligned")
    y = labels["target"].to_numpy(dtype=np.int8)
    strata = tuple(c for c in strata if c in labels.columns)
    masks = panel_masks(labels, strata)
    primary_score = detection_score(scores[primary].to_numpy())
    bootstrap = joint_bootstrap(
        y, primary_score, [cluster_key(v) for v in labels["video_id"]],
        masks, n_resamples=n_resamples, seed=seed, scheme=scheme,
        batch_size=batch_size)
    panels = {}
    for name, mask in masks.items():
        ym = y[mask]
        columns = {c: score_panel(ym, scores[c].to_numpy()[mask], c,
                                  require_complete=(c == primary))
                   for c in score_columns}
        head = columns[primary]
        if not head["defined"]:
            raise ValueError(f"{name}: the primary score is undefined")
        panels[name] = {
            "population": {"n_real": int((ym == 0).sum()),
                           "n_fake": int((ym == 1).sum())},
            "roc_auc": head["roc_auc"],
            "operating_points": head["operating_points"],
            "uncertainty": {metric: interval(values) for metric, values
                            in bootstrap.draws[name].items()},
            "other_roc_auc": {c: p["roc_auc"] for c, p in columns.items()
                              if c != primary and p["defined"]},
            "exact_rank_rates": exact_rank_rates(
                ym, scores[exact_rank].to_numpy()[mask]),
            "score_panels": columns,
        }
    overall = panels.pop("overall")
    return {
        "primary_score": primary, "exact_rank_score": exact_rank,
        "score_columns": list(score_columns), "statistic": f"-log({primary})",
        "overall": overall, "strata": panels,
        "bootstrap": {"statistic": f"ROC-AUC and average precision of -log({primary})",
                      "unit": "speaker cluster via cast.eval.bootstrap.speaker_cluster_key",
                      "n_clusters": bootstrap.n_clusters,
                      "n_resamples": bootstrap.n_resamples, "seed": bootstrap.seed,
                      "resampling": bootstrap.scheme, "batch_size": bootstrap.batch_size,
                      "joint_draws": True, "confidence_level": 0.95,
                      "interval_type": "percentile", "percentiles": [2.5, 97.5],
                      "numpy_method": "linear", "sd_ddof": 0},
    }
