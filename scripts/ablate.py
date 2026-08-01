"""Ablation harness: does each new feature-group treatment help or hurt accuracy?

Trains the same tuned-XGBoost pipeline (`train.tune_and_train`) on several feature-group
combinations and compares them, per horizon, on the threshold-aware metrics introduced in
item C (overall MAE/RMSE, below-threshold MAE, exceedance recall/precision/fbeta). Models
trained here are never persisted to `models/model_{h}h.ubj` (`save_model=False`), so this
script cannot clobber the production models produced by `scripts/train.py`.

Usage:
    python scripts/ablate.py [--trials 20] [--horizons 1 4 12]

Writes data/processed/ablation_report.json (full nested results) and
data/processed/ablation_report.csv (one row per combination x horizon).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from air_quality_forecast import config, features  # noqa: E402
import train  # noqa: E402

# Feature-group combinations to compare. Extensible: add a (name, groups) tuple to try
# a new combination without touching the rest of the harness. "base" (lag/roll/calendar
# features) is the reference every other combination is diffed against.
COMBINATIONS: list[tuple[str, list[str]]] = [
    ("base", ["base"]),
    ("base+precip", ["base", "precip"]),
    ("base+fft", ["base", "fft"]),
    ("base+wavelet", ["base", "wavelet"]),
    ("base+denoise", ["base", "denoise"]),
    ("base+all", ["base", "precip", "fft", "wavelet", "denoise"]),
]

# Metric columns pulled out of `model_threshold_report` for the comparison table/CSV, and
# whether a lower value is better (used to label deltas as improving/worsening vs base).
METRIC_COLUMNS: list[tuple[str, bool]] = [
    ("overall_rmse", True),
    ("overall_mae", True),
    ("below_threshold_mae", True),
    ("exceedance_recall", False),
    ("exceedance_precision", False),
    ("exceedance_fbeta", False),
]


def extract_metrics(n_features: int, result: dict) -> dict:
    """Pull the comparison metrics out of a `tune_and_train` result dict."""
    report = result["model_threshold_report"]
    exceedance = report["exceedance"]
    return {
        "n_features": n_features,
        "overall_rmse": report["overall_rmse"],
        "overall_mae": report["overall_mae"],
        "below_threshold_mae": report["below_threshold_mae"],
        "exceedance_recall": exceedance["recall"],
        "exceedance_precision": exceedance["precision"],
        "exceedance_fbeta": exceedance["fbeta"],
    }


def print_comparison_table(horizon: int, metrics_by_combo: dict[str, dict]) -> None:
    """Print each combination's metrics and their delta vs the `base` combination.

    For lower-is-better metrics (rmse/mae/below_threshold_mae) a negative delta means the
    combination improved on base; for higher-is-better metrics (recall/precision/fbeta) a
    positive delta means improvement.
    """
    base = metrics_by_combo["base"]
    print(f"[horizon {horizon}h] comparison vs base (n_features={base['n_features']})")
    header = f"  {'combination':<14}{'n_feat':>8}"
    for col, _ in METRIC_COLUMNS:
        header += f"{col:>22}"
    print(header)
    for name, m in metrics_by_combo.items():
        row = f"  {name:<14}{m['n_features']:>8}"
        for col, lower_is_better in METRIC_COLUMNS:
            value = m[col]
            if name == "base":
                row += f"{value:>22.4f}"
            else:
                delta = value - base[col]
                better = delta < 0 if lower_is_better else delta > 0
                arrow = "v" if better else ("^" if delta != 0 else "=")
                row += f"{value:>14.4f}({arrow}{delta:+.4f})"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=config.FORECAST_HORIZONS_H
    )
    args = parser.parse_args()

    df = train.load_dataset()

    full_report: dict = {}
    csv_rows: list[dict] = []

    for horizon in args.horizons:
        metrics_by_combo: dict[str, dict] = {}
        full_report[f"{horizon}h"] = {}

        for combo_name, groups in COMBINATIONS:
            feat_cols = features.select_feature_groups(df, groups)
            result = train.tune_and_train(
                horizon, df, feat_cols, args.trials, save_model=False
            )
            combo_metrics = extract_metrics(len(feat_cols), result)
            metrics_by_combo[combo_name] = combo_metrics

            full_report[f"{horizon}h"][combo_name] = {
                "groups": groups,
                "n_features": len(feat_cols),
                "result": result,
            }
            csv_rows.append(
                {
                    "horizon_h": horizon,
                    "combination": combo_name,
                    "groups": "+".join(groups),
                    **combo_metrics,
                }
            )

        print_comparison_table(horizon, metrics_by_combo)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.PROCESSED_DIR / "ablation_report.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"[saved] {json_path}")

    csv_path = config.PROCESSED_DIR / "ablation_report.csv"
    fieldnames = ["horizon_h", "combination", "groups", "n_features"] + [
        col for col, _ in METRIC_COLUMNS
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
