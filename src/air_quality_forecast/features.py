"""Feature engineering shared by training and inference/backtest (train/serve parity).

`build_features` and `add_targets` are the only two functions either the training
pipeline or the dashboard are allowed to call to go from a clean hourly series to a
model-ready table. Never reimplement this logic in a second place.
"""
import warnings

import numpy as np
import pandas as pd
import pywt
from numpy.lib.stride_tricks import sliding_window_view

from . import config
from .splits import TRAIN_END


def _trailing_windows(shifted_values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build trailing causal windows of length `window` ending at each index.

    `shifted_values` must already be a single site's shift(1)-lagged pm25 series, so
    `windows[i]` covers strictly [t-1-window, t-1] relative to row i's timestamp t — the
    current reading is never included, matching the no-leakage rule used elsewhere in this
    module. Returns `(windows, full_mask)`: `windows` has shape (len(shifted_values), window);
    `full_mask[i]` is False for the first `window - 1` rows of the site, which do not yet
    have a full trailing history and must be reported as NaN by the caller.
    """
    n = len(shifted_values)
    padded = np.concatenate([np.full(window - 1, np.nan), shifted_values.astype(float)])
    windows = sliding_window_view(padded, window)
    full_mask = np.arange(n) >= (window - 1)
    return windows, full_mask


def _impute_windows(windows: np.ndarray) -> np.ndarray:
    """Fill NaNs inside each window with that window's own mean (0.0 if fully NaN).

    Imputation is per-row (per-window) and only ever looks at values already inside the
    trailing window, so it cannot introduce any future information.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        # All-NaN windows (rows without a full trailing history yet) trigger numpy's
        # "Mean of empty slice" warning; those rows are masked out by the caller anyway.
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(windows, axis=1)
    means = np.where(np.isnan(means), 0.0, means)
    nan_mask = np.isnan(windows)
    return np.where(nan_mask, means[:, None], windows)


def _fft_features(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Batched rfft band-energy / dominant-frequency / spectral-entropy features.

    `windows` is (n_rows, SPECTRAL_WINDOW_H) of already-imputed trailing pm25 windows;
    the transform is applied along axis=1 (one FFT per row, all rows at once).
    """
    n_rows, width = windows.shape
    spectrum = np.fft.rfft(windows, axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(width)  # sample spacing = 1h, so freqs are in cycles/hour

    out: dict[str, np.ndarray] = {}
    edges = config.FFT_BAND_EDGES_CPH
    for name, lo, hi in zip(config.FFT_BAND_NAMES, edges[:-1], edges[1:]):
        band_mask = (freqs >= lo) & (freqs < hi)
        out[f"fft_energy_{name}"] = power[:, band_mask].sum(axis=1)

    # Dominant frequency and spectral entropy exclude the DC (index 0) bin, which reflects
    # the window's mean level rather than any periodic structure.
    ac_power = power[:, 1:]
    total = ac_power.sum(axis=1)
    total_safe = np.where(total <= 0, 1.0, total)
    dom_idx = np.argmax(ac_power, axis=1)
    out["fft_dom_freq_idx"] = (dom_idx + 1).astype(float)
    out["fft_dom_freq_mag"] = ac_power[np.arange(n_rows), dom_idx]

    probs = ac_power / total_safe[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    max_entropy = np.log(ac_power.shape[1])
    out["fft_spectral_entropy"] = entropy / max_entropy
    return out


def _wavelet_and_denoise_features(windows: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Batched wavelet-energy features plus a causal wavelet-denoised value per row.

    `windows` is (n_rows, SPECTRAL_WINDOW_H) of already-imputed trailing pm25 windows.
    `pywt.wavedec(..., axis=1)` decomposes all rows in one batched call (no per-row loop).
    Energy is reported per decomposition level (approximation + each detail level).

    For denoising, detail coefficients are soft-thresholded with the VisuShrink universal
    threshold (noise sigma estimated per row from the finest detail level via MAD), then the
    signal is reconstructed with `pywt.waverec`. Only the LAST reconstructed sample (which
    corresponds to t-1, the most recent value in the window) is kept, so the result is a
    strictly causal, denoised estimate of the previous pm25 reading for every row.
    """
    with warnings.catch_warnings():
        # WAVELET_LEVELS can exceed pywt's recommended max level for this window length;
        # that only means boundary effects at the coarsest level, not incorrect output.
        warnings.simplefilter("ignore", UserWarning)
        coeffs = pywt.wavedec(windows, config.WAVELET, level=config.WAVELET_LEVELS, axis=1)

    approx, *details = coeffs  # details ordered coarsest (level L) -> finest (level 1)
    n_details = len(details)

    wav_out: dict[str, np.ndarray] = {"wav_energy_approx": np.sum(approx**2, axis=1)}
    for i, det in enumerate(details):
        level_label = n_details - i  # first entry is the coarsest detail = level n_details
        wav_out[f"wav_energy_detail_{level_label}"] = np.sum(det**2, axis=1)

    finest_detail = details[-1]
    with np.errstate(invalid="ignore"):
        sigma = np.median(np.abs(finest_detail), axis=1) / 0.6745
    width = windows.shape[1]
    threshold = sigma * np.sqrt(2 * np.log(width))
    with np.errstate(invalid="ignore", divide="ignore"):
        # pywt's soft-threshold formula divides by the coefficient magnitude, which is
        # exactly 0 for exact-zero detail coefficients (e.g. a fully flat window); pywt
        # already resolves that 0/0 back to 0, this just silences the transient warning.
        thresholded = [approx] + [
            pywt.threshold(det, threshold[:, None], mode="soft") for det in details
        ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reconstructed = pywt.waverec(thresholded, config.WAVELET, axis=1)
    denoised_last = reconstructed[:, : windows.shape[1]][:, -1]
    return wav_out, denoised_last


def _add_spectral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal FFT (`fft_`), wavelet (`wav_`), and denoising (`den_`) features per site.

    All features are derived from trailing SPECTRAL_WINDOW_H-hour windows of `grp.shift(1)`
    pm25 (see `_trailing_windows`), so — like the rolling stats in `build_features` — they
    can only ever depend on pm25 values at or before t-1. Rows without a full trailing
    window (the first SPECTRAL_WINDOW_H - 1 rows of each site) are NaN.
    """
    window = config.SPECTRAL_WINDOW_H
    fft_cols: dict[str, list[np.ndarray]] = {}
    wav_cols: dict[str, list[np.ndarray]] = {}
    den_now_parts: list[np.ndarray] = []
    site_order: list[np.ndarray] = []

    for site_id, site_df in df.groupby("site_id", sort=False):
        idx = site_df.index.to_numpy()
        site_order.append(idx)
        shifted = site_df[config.TARGET_COL].shift(1).to_numpy()

        windows, full_mask = _trailing_windows(shifted, window)
        imputed = _impute_windows(windows)

        fft_feats = _fft_features(imputed)
        wav_feats, den_now = _wavelet_and_denoise_features(imputed)

        for name, values in fft_feats.items():
            values = np.where(full_mask, values, np.nan)
            fft_cols.setdefault(name, []).append(values)
        for name, values in wav_feats.items():
            values = np.where(full_mask, values, np.nan)
            wav_cols.setdefault(name, []).append(values)

        den_now_parts.append(np.where(full_mask, den_now, np.nan))

    order = np.concatenate(site_order)
    sort_pos = np.argsort(order)

    for name, parts in fft_cols.items():
        df[name] = np.concatenate(parts)[sort_pos]
    for name, parts in wav_cols.items():
        df[name] = np.concatenate(parts)[sort_pos]
    df["den_now"] = np.concatenate(den_now_parts)[sort_pos]

    # A couple of lag/rolling features derived from the denoised series itself, reusing
    # existing LAG_HOURS / ROLLING_WINDOWS_H values so the column count stays modest. Each
    # is strictly causal because den_now is itself already a t-1-only quantity.
    den_grp = df.groupby("site_id")["den_now"]
    df["den_lag_1h"] = den_grp.shift(1)
    df["den_lag_24h"] = den_grp.shift(24)
    df["den_roll_24h_mean"] = (
        den_grp.rolling(24, min_periods=12).mean().reset_index(level=0, drop=True)
    )
    return df


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year harmonic (`seas_doy_`) and climatological-anomaly (`seas_`) features.

    Day-of-year harmonics -- `seas_doy_sin_{k}`/`seas_doy_cos_{k}` for k = 1..
    config.SEASONAL_HARMONICS, using doy = df["dt"].dt.dayofyear over a 365.25-day year --
    give a continuous annual cycle free of the month-boundary discontinuities in month_sin/
    cos, and (with multiple harmonics) can represent a non-sinusoidal annual shape. That
    matters here because California PM2.5 has two distinct seasonal peaks (a sharp winter
    Central Valley inversion peak and a separately-shaped wildfire-season peak) that a single
    sin/cos pair cannot fit. month_sin/cos is left untouched (still part of the "base" group)
    so existing recorded results stay comparable.

    Climatological anomaly -- expresses how unusual the most recent reading is for this
    site, at this time of year, at this hour, which is far more informative than the raw
    level (e.g. 20 ug/m3 is anomalous on the coast in summer, routine in Fresno in winter):
      seas_climatology     -- mean pm25 for this row's (site_id, calendar month, hour-of-day)
                               cell. Cell definition = 5 sites x 12 months x 24 hours = 1440
                               cells over the training rows (~90 samples/cell at hourly
                               resolution over ~2 training years) -- dense enough to be
                               stable. Day-of-year cells (5 x 365 x 24, ~3 samples/cell) were
                               rejected as far too sparse.
      seas_anomaly_1h       -- shift(1) pm25 minus seas_climatology.
      seas_anomaly_ratio    -- shift(1) pm25 divided by seas_climatology (NaN whenever the
                               climatology cell is NaN or <= 0, since concentrations are
                               non-negative and a ratio against a non-positive/undefined
                               baseline is meaningless).
      seas_anomaly_24h_mean -- trailing 24h mean of seas_anomaly_1h. No extra shift is
                               applied here (same pattern as den_roll_24h_mean in
                               `_add_spectral_features`): seas_anomaly_1h is already a
                               t-1-only quantity for every row, so rolling over it directly
                               stays strictly backward-looking.

    Leakage rule (both parts mandatory):
    (a) The climatology is estimated using ONLY rows with dt < splits.TRAIN_END, never the
        full dataset, so no validation/test-period information leaks into it. TRAIN_END is a
        fixed constant, so the dashboard/inference path recomputes the identical statistic
        from the same historical rows at serve time -- train/serve parity holds without
        needing to ship a separately-fitted lookup table.
    (b) The anomaly columns are always computed against grp.shift(1) pm25 -- the same lagged
        series the rolling stats use -- never the current row's pm25, so they cannot leak
        the target.

    Climatology cells with zero training-period rows for a given (site, month, hour) yield
    NaN for seas_climatology (and therefore for the two anomaly columns derived from it); no
    fallback to a global mean is used, so as not to silently blend unrelated regimes.
    """
    dt = df["dt"]
    doy = dt.dt.dayofyear
    for k in range(1, config.SEASONAL_HARMONICS + 1):
        df[f"seas_doy_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"seas_doy_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    month = dt.dt.month
    hour = dt.dt.hour
    train_mask = (dt < TRAIN_END).to_numpy()

    climatology = (
        df.loc[train_mask]
        .assign(_month=month.to_numpy()[train_mask], _hour=hour.to_numpy()[train_mask])
        .groupby(["site_id", "_month", "_hour"])[config.TARGET_COL]
        .mean()
    )
    cell_index = pd.MultiIndex.from_arrays(
        [df["site_id"], month, hour], names=["site_id", "_month", "_hour"]
    )
    df["seas_climatology"] = climatology.reindex(cell_index).to_numpy()

    shifted = df.groupby("site_id")[config.TARGET_COL].shift(1)
    clim = df["seas_climatology"]
    df["seas_anomaly_1h"] = shifted - clim
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = shifted / clim
    df["seas_anomaly_ratio"] = np.where(clim > 0, ratio, np.nan)

    anom_grp = df.groupby("site_id")["seas_anomaly_1h"]
    df["seas_anomaly_24h_mean"] = (
        anom_grp.rolling(24, min_periods=12).mean().reset_index(level=0, drop=True)
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling-window, spectral, and cyclical calendar features.

    `df` must have columns [site_id, dt, pm25], one row per site per hour, sorted by
    (site_id, dt). All lag/rolling/spectral features are strictly backward-looking (based
    on shift >= 1 series) so nothing here can leak future information into a training row.
    See `_add_spectral_features` for the FFT (`fft_`), wavelet (`wav_`), and denoising
    (`den_`) features.
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

    df = _add_spectral_features(df)

    hour = df["dt"].dt.hour
    dow = df["dt"].dt.dayofweek
    month = df["dt"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df = _add_seasonal_features(df)

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


# Column membership per feature group: (prefixes, exact column names). Used by
# `select_feature_groups` to let callers (e.g. model-comparison experiments) opt into a
# subset of `build_features` output without hand-listing column names.
_FEATURE_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "base": (("lag_", "roll_", "hour_", "dow_", "month_"), ("site_id",)),
    "precip": (("precip_", "rh_"), ("precipitation", "relative_humidity_2m")),
    "fft": (("fft_",), ()),
    "wavelet": (("wav_",), ()),
    "denoise": (("den_",), ()),
    "seasonal": (("seas_",), ()),
}


def select_feature_groups(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """Return the subset of `feature_columns(df)` belonging to the requested groups.

    Groups are matched by column prefix (plus a few exact names for "base"/"precip"):
    "base" -> lag_, roll_, hour_, dow_, month_, site_id; "precip" -> precip_, rh_,
    precipitation, relative_humidity_2m; "fft" -> fft_; "wavelet" -> wav_; "denoise" -> den_;
    "seasonal" -> seas_. Raises ValueError for an unrecognized group name.
    """
    unknown = set(groups) - _FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {sorted(unknown)}. "
            f"Valid groups: {sorted(_FEATURE_GROUPS)}"
        )

    available = feature_columns(df)
    selected: list[str] = []
    for group in groups:
        prefixes, exact_names = _FEATURE_GROUPS[group]
        for col in available:
            if col in exact_names or col.startswith(prefixes):
                if col not in selected:
                    selected.append(col)
    return selected
