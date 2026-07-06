"""Asymmetric sample weighting: missing a pollution spike is operationally worse than a
false alarm (a health advisory that didn't need to fire is cheap; a spike nobody warned
about is not). Weight is derived from the training split's own target distribution, not
a hardcoded constant, so it adapts per-site/per-horizon rather than encoding one fixed
domain threshold.
"""
import numpy as np
import pandas as pd

SPIKE_QUANTILE = 0.75
SPIKE_WEIGHT_ALPHA = 2.0


def spike_threshold(y_train: pd.Series) -> float:
    return float(y_train.quantile(SPIKE_QUANTILE))


def sample_weights(y: pd.Series, threshold: float, alpha: float = SPIKE_WEIGHT_ALPHA) -> np.ndarray:
    excess = np.clip(y.to_numpy() - threshold, a_min=0, a_max=None)
    scale = max(threshold, 1e-6)
    return 1.0 + alpha * (excess / scale)
