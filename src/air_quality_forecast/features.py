"""Feature engineering shared by training and inference/backtest (train/serve parity).

`build_features` and `add_targets` are the only two functions either the training
pipeline or the dashboard are allowed to call to go from a clean hourly series to a
model-ready table. Never reimplement this logic in a second place.
"""
import numpy as np
import pandas as pd

from . import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling-window, and cyclical calendar features.

    `df` must have columns [site_id, dt, pm25], one row per site per hour, sorted by
    (site_id, dt). All lag/rolling features are strictly backward-looking (shift >= 1)
    so nothing here can leak future information into a training row.
    """
    df = df.sort_values(["site_id", "dt"]).reset_index(drop=True)
    grp = df.groupby("site_id")[config.TARGET_COL]

    for lag in config.LAG_HOURS:
        df[f"lag_{lag}h"] = grp.shift(lag)

    # Rolling stats computed on values shifted by 1h first, so the window for row t
    # only ever covers [t-1-window, t-1] — never includes the current reading.
    shifted = grp.shift(1)
    for window in config.ROLLING_WINDOWS_H:
        roll = shifted.groupby(df["site_id"]).rolling(window, min_periods=max(2, window // 2))
        df[f"roll_{window}h_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"roll_{window}h_std"] = roll.std().reset_index(level=0, drop=True)
        df[f"roll_{window}h_max"] = roll.max().reset_index(level=0, drop=True)

    if "precipitation" in df.columns and "relative_humidity_2m" in df.columns:
        # Coerce defensively: weather columns may arrive as object/pd.NA dtype (e.g. when no
        # weather has been downloaded yet), which pandas rolling() cannot aggregate directly.
        df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
        df["relative_humidity_2m"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")

        precip_grp = df.groupby("site_id")["precipitation"]
        for window in config.PRECIP_WINDOWS_H:
            # Precipitation is exogenous observed data (not the pm25 target), so the trailing
            # window through the current hour is legitimate — no shift(1) needed here, unlike
            # the pm25 rolling stats above.
            roll = precip_grp.rolling(window, min_periods=max(2, window // 2))
            df[f"precip_{window}h_sum"] = roll.sum().reset_index(level=0, drop=True)

        rh_grp = df.groupby("site_id")["relative_humidity_2m"]
        for window in config.PRECIP_WINDOWS_H:
            roll = rh_grp.rolling(window, min_periods=max(2, window // 2))
            df[f"rh_{window}h_mean"] = roll.mean().reset_index(level=0, drop=True)
        df["rh_now"] = df["relative_humidity_2m"]

    hour = df["dt"].dt.hour
    dow = df["dt"].dt.dayofweek
    month = df["dt"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["site_id"] = df["site_id"].astype("category")
    return df


def add_targets(df: pd.DataFrame, horizons_h=None) -> pd.DataFrame:
    """Add target_{h}h columns: the pm25 reading h hours after the current row."""
    horizons_h = horizons_h or config.FORECAST_HORIZONS_H
    grp = df.groupby("site_id")[config.TARGET_COL]
    for h in horizons_h:
        df[f"target_{h}h"] = grp.shift(-h)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"dt", config.TARGET_COL} | {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude]
