"""Train one Optuna-tuned XGBoost regressor per forecast horizon.

Usage:
    python scripts/train.py [--trials 40]

Saves, per horizon: models/model_{h}h.json, and a combined models/metrics.json
comparing the tuned model against a persistence baseline and a rolling-mean baseline.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_quality_forecast import config, features, losses, metrics, splits  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "features.parquet")


def prepare_xy(df: pd.DataFrame, horizon: int, feat_cols: list[str]):
    target_col = f"target_{horizon}h"
    valid = df.dropna(subset=[target_col, config.TARGET_COL])
    return valid[feat_cols], valid[target_col], valid[config.TARGET_COL], valid["dt"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def baseline_metrics(y_true: pd.Series, current_pm25: pd.Series) -> dict:
    """Persistence baseline: predict the current reading holds constant over the horizon."""
    return {
        "mae": float(mean_absolute_error(y_true, current_pm25)),
        "rmse": rmse(y_true, current_pm25),
    }


def select_pareto_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Pick the trial to deploy from a multi-objective study's Pareto front.

    `study.best_trials` is the Pareto front for objectives
    (below_threshold_mae [minimize], exceedance_fbeta [maximize]). Among the front's
    trials whose below_threshold_mae is within `config.PARETO_MAE_TOLERANCE` of the
    front's best (lowest) below_threshold_mae, select the one with the highest
    exceedance_fbeta, breaking ties by lower below_threshold_mae. This favors detection
    performance without accepting an unbounded regression-accuracy cost in the calm
    regime.
    """
    front = study.best_trials
    best_mae = min(t.values[0] for t in front)
    cutoff = best_mae * (1 + config.PARETO_MAE_TOLERANCE)
    eligible = [t for t in front if t.values[0] <= cutoff]
    return max(eligible, key=lambda t: (t.values[1], -t.values[0]))


def tune_and_train(
    horizon: int,
    df: pd.DataFrame,
    feat_cols: list[str],
    n_trials: int,
    save_model: bool = True,
) -> dict:
    train_df, val_df, test_df = splits.split(df)

    X_train, y_train, cur_train, _ = prepare_xy(train_df, horizon, feat_cols)
    X_val, y_val, cur_val, _ = prepare_xy(val_df, horizon, feat_cols)
    X_test, y_test, cur_test, _ = prepare_xy(test_df, horizon, feat_cols)

    health_threshold = config.HEALTH_THRESHOLD_UGM3
    w_train = losses.sample_weights(y_train, health_threshold)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-1, 20, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        model = xgb.XGBRegressor(
            n_estimators=2000,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=50,
            eval_metric="rmse",
            **params,
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        below_mae_val = metrics.below_threshold_mae(y_val, preds, health_threshold)
        fbeta_val = metrics.exceedance_scores(
            y_val, preds, health_threshold, config.DETECTION_FBETA
        )["fbeta"]
        return below_mae_val, fbeta_val

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    selected_trial = select_pareto_trial(study)
    best_params = selected_trial.params

    # Refit on train+val with the selected params, evaluate once on the held-out test split.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    w_trainval = losses.sample_weights(y_trainval, health_threshold)

    # Re-derive a good n_estimators via early stopping on a held-back tail of train+val.
    cutoff = int(len(X_trainval) * 0.9)
    final_model = xgb.XGBRegressor(
        n_estimators=2000,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        eval_metric="rmse",
        **best_params,
    )
    final_model.fit(
        X_trainval.iloc[:cutoff], y_trainval.iloc[:cutoff],
        sample_weight=w_trainval[:cutoff],
        eval_set=[(X_trainval.iloc[cutoff:], y_trainval.iloc[cutoff:])],
        verbose=False,
    )

    test_preds = final_model.predict(X_test)
    model_metrics = {
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "rmse": rmse(y_test, test_preds),
    }
    persistence = baseline_metrics(y_test, cur_test)

    model_threshold_report = metrics.threshold_report(
        y_test, test_preds, health_threshold, config.DETECTION_FBETA
    )
    persistence_threshold_report = metrics.threshold_report(
        y_test, cur_test, health_threshold, config.DETECTION_FBETA
    )

    pareto_front = [
        {
            "below_mae": t.values[0],
            "fbeta": t.values[1],
            "params": t.params,
        }
        for t in study.best_trials
    ]
    selected_summary = {
        "below_mae": selected_trial.values[0],
        "fbeta": selected_trial.values[1],
        "params": selected_trial.params,
        "trial_number": selected_trial.number,
    }

    if save_model:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        final_model.save_model(config.MODELS_DIR / f"model_{horizon}h.ubj")

    print(
        f"[horizon {horizon}h] model RMSE={model_metrics['rmse']:.3f} MAE={model_metrics['mae']:.3f} "
        f"| persistence RMSE={persistence['rmse']:.3f} MAE={persistence['mae']:.3f} "
        f"| n_trials={n_trials} pareto_front={len(pareto_front)} "
        f"selected_trial={selected_trial.number} best_params={best_params}"
    )

    return {
        "horizon_h": horizon,
        "health_threshold_ugm3": health_threshold,
        "best_params": best_params,
        "n_boost_rounds": final_model.best_iteration,
        "model": model_metrics,
        "baseline_persistence": persistence,
        "model_threshold_report": model_threshold_report,
        "baseline_persistence_threshold_report": persistence_threshold_report,
        "pareto_front": pareto_front,
        "selected_trial": selected_summary,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()

    df = load_dataset()
    feat_cols = features.feature_columns(df)

    all_metrics = {}
    for horizon in config.FORECAST_HORIZONS_H:
        all_metrics[f"{horizon}h"] = tune_and_train(horizon, df, feat_cols, args.trials)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[saved] {config.MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
