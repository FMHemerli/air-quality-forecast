"""Threshold-aware evaluation metrics for the health-exceedance forecasting objective.

The forecast must serve two different goals depending on the regime of the true value
relative to a health threshold `T` (µg/m3): when the true value is at or below `T`
(calm regime), regression accuracy is what matters; when the true value exceeds `T`,
correctly flagging the exceedance matters more than the exact magnitude of the error.
These functions are pure numpy/pandas (no XGBoost dependency) so they can be reused
identically by the training objective, the final test-set report, and any downstream
serving/monitoring code (train/serve parity).
"""
import numpy as np
import pandas as pd


def _as_array(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    return np.asarray(x)


def below_threshold_mae(y_true, y_pred, threshold: float) -> float:
    """Mean absolute error restricted to rows where the true value is <= threshold.

    This is the "calm regime" error: how accurate the forecast is when no health
    exceedance is actually occurring. Returns nan if no such rows exist.
    """
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    mask = y_true <= threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def exceedance_scores(y_true, y_pred, threshold: float, beta: float) -> dict:
    """Binary exceedance-detection scores treating `y_true > threshold` as positive.

    Predicted-positive is `y_pred > threshold`. Recall, precision, and fbeta are each
    defined as 0.0 when their denominator is zero (e.g. no true exceedances in the
    sample, or the model never predicts an exceedance) rather than raising, since the
    training objective must be able to evaluate every trial including degenerate ones.
    """
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)

    actual_pos = y_true > threshold
    pred_pos = y_pred > threshold

    tp = int(np.sum(actual_pos & pred_pos))
    fp = int(np.sum(~actual_pos & pred_pos))
    fn = int(np.sum(actual_pos & ~pred_pos))
    tn = int(np.sum(~actual_pos & ~pred_pos))
    n_positive = int(np.sum(actual_pos))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    beta_sq = beta * beta
    fbeta_denom = (beta_sq * precision) + recall
    fbeta = (
        (1 + beta_sq) * precision * recall / fbeta_denom
        if fbeta_denom > 0
        else 0.0
    )

    return {
        "recall": float(recall),
        "precision": float(precision),
        "fbeta": float(fbeta),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_positive": n_positive,
    }


def threshold_report(y_true, y_pred, threshold: float, beta: float) -> dict:
    """Combine below-threshold MAE, exceedance detection scores, and overall MAE/RMSE.

    Convenience wrapper meant for writing a single threshold-aware evaluation block
    into metrics.json (e.g. one per horizon, for both the model and the persistence
    baseline).
    """
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)

    overall_mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    overall_rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    return {
        "threshold_ugm3": float(threshold),
        "below_threshold_mae": below_threshold_mae(y_true_arr, y_pred_arr, threshold),
        "exceedance": exceedance_scores(y_true_arr, y_pred_arr, threshold, beta),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
    }
