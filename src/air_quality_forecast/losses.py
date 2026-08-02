"""Asymmetric sample weighting: missing a pollution spike is operationally worse than a
false alarm (a health advisory that didn't need to fire is cheap; a spike nobody warned
about is not).

The weight is anchored on a fixed configured constant (`config.SPIKE_WEIGHT_ANCHOR_UGM3`),
supplied by the caller. That anchor is deliberately NOT the detection threshold
(`config.HEALTH_THRESHOLD_UGM3`): weighting and detection answer different questions, and
tying them together made the weighting collapse when detection moved to 35 µg/m³ — only 2.0%
of training rows would be upweighted, at a mean weight of 1.014, versus 12.2% and 1.180 at
an anchor of 15. See the comments on both constants in config.py.
"""
import numpy as np
import pandas as pd

SPIKE_WEIGHT_ALPHA = 2.0


def sample_weights(y: pd.Series, threshold: float, alpha: float = SPIKE_WEIGHT_ALPHA) -> np.ndarray:
    excess = np.clip(y.to_numpy() - threshold, a_min=0, a_max=None)
    scale = max(threshold, 1e-6)
    return 1.0 + alpha * (excess / scale)
