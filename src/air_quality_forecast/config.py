"""Shared configuration: site list, forecast horizons, paths.

Used by both the training pipeline and the dashboard so the two never drift apart
(train/serve parity).
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

YEARS = [2022, 2023, 2024]
STATE_CODE = "06"  # California

# County_Code-Site_Num identifiers, chosen for a mix of pollution regimes:
# Fresno and Kings (Central Valley, ag burning + wildfire smoke, high spikes),
# Alameda and Monterey (coastal Bay Area / Central Coast, cleaner baseline),
# Ventura (South Coast, moderate urban/coastal mix).
SITES = {
    "019-5025": "Fresno",
    "031-0004": "Kings (Corcoran)",
    "001-0009": "Alameda (Oakland)",
    "053-1003": "Monterey (Salinas)",
    "111-1004": "Ventura",
}

FORECAST_HORIZONS_H = [1, 4, 12]

# Feature construction window sizes (in hours), all backward-looking to avoid leakage.
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS_H = [3, 6, 12, 24]

# Number of shortest LAG_HOURS lag columns considered "core": the minimum recent history a
# row needs to be meaningfully predictable at all. Used by
# features.core_lag_columns/dashboard/app.py to decide which rows have enough signal to
# forecast, without requiring every (possibly NaN) feature column to be populated.
CORE_LAG_COUNT = 3

# Backward-looking precipitation aggregation windows (in hours), used by weather features.
PRECIP_WINDOWS_H = [3, 6, 12, 24, 48]

# Trailing causal window (in hours) used by the FFT / wavelet / denoising signal features.
# Each window covers [t-1-SPECTRAL_WINDOW_H, t-1] of shifted pm25 — never the current
# reading — so it obeys the same no-leakage rule as the rolling stats above.
SPECTRAL_WINDOW_H = 48

# Wavelet family and decomposition depth shared by the wavelet-energy and causal-denoising
# features (train/serve parity: both use pywt.wavedec with these exact settings).
WAVELET = "db4"
WAVELET_LEVELS = 3

# Band edges (cycles/hour) splitting the rfft spectrum of a SPECTRAL_WINDOW_H-hour window
# into three bands: "low" (slow, multi-day trend), "diurnal" (~24h cycle, centered on
# 1/24 cph), and "high" (sub-12h fluctuations / noise). Nyquist for hourly data is 0.5 cph.
FFT_BAND_NAMES = ["low", "diurnal", "high"]
FFT_BAND_EDGES_CPH = [0.0, 1 / 36, 1 / 12, 0.5]

# Number of sin/cos harmonic pairs used for the day-of-year seasonal cycle
# (features.seas_doy_sin_k/seas_doy_cos_k). A single harmonic (like month_sin/cos) can only
# represent a pure sinusoid; California PM2.5 has a non-sinusoidal annual shape (a sharp
# winter Central Valley inversion peak plus a separate, differently-shaped wildfire-season
# peak), which needs the extra harmonics to be approximated well.
SEASONAL_HARMONICS = 3

# Open-Meteo historical weather API (no key required).
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TARGET_COL = "pm25"

# Health-exceedance threshold (µg/m3). Anchors DETECTION ONLY: the multi-objective Optuna
# metric (metrics.exceedance_scores), the Pareto selection rule, and the primary threshold
# report written to models/metrics.json. It deliberately does NOT anchor the training sample
# weighting -- see SPIKE_WEIGHT_ANCHOR_UGM3 below.
#
# 35.0 sits in the US AQI "Unhealthy for Sensitive Groups" region. It is a round-number
# stand-in, not the official breakpoint: the 2024 AQI revision puts that band at 35.5-55.4
# µg/m3, and as a 24-hour average. Moving here from the WHO 15 µg/m3 guideline reduces one of
# the three objections in docs/24h-guideline-hourly-forecast.md -- exceedance drops from
# ~13% of hours to ~2%, so F-beta is far less inflated by base rate -- but does NOT fix the
# averaging-window mismatch, since 35 is a 24h standard applied hourly here too.
HEALTH_THRESHOLD_UGM3 = 35.0

# Additional thresholds (µg/m3) evaluated and reported alongside the primary one. Pure
# post-hoc readout on frozen predictions: these never touch training, tuning, or model
# selection. 15.0 is the WHO 24h guideline, kept so the dashboard can compare the two and so
# the numbers published before the migration to 35 stay reproducible.
SECONDARY_HEALTH_THRESHOLDS_UGM3 = [15.0]

# Anchor (µg/m3) for the asymmetric training sample weighting (losses.sample_weights),
# deliberately decoupled from HEALTH_THRESHOLD_UGM3. Measured on the training split: an
# anchor of 15 upweights 12.2% of rows (mean weight 1.180), while an anchor of 35 reaches
# only 2.0% (mean weight 1.014) -- near enough to an identity transform that the "missing a
# spike costs more than a false alarm" mechanism would stop doing meaningful work. Detection
# moved to 35; the weighting stays where it still bites. Do not collapse the two constants
# back into one.
SPIKE_WEIGHT_ANCHOR_UGM3 = 15.0

# Human-readable provenance for each threshold, shared by the dashboard (axis annotation,
# selector labels) and the docs, so the standard a line refers to is never hardcoded next to
# a number it does not match.
THRESHOLD_LABELS = {
    35.0: "US AQI Unhealthy for Sensitive Groups",
    15.0: "WHO 24h guideline",
}

# Beta for the F-beta exceedance-detection metric (metrics.exceedance_scores). beta > 1
# weights recall over precision, reflecting that missing a health exceedance is worse
# than a false alarm.
DETECTION_FBETA = 2.0

# Pareto-front selection tolerance (see scripts/train.py select_pareto_trial): among
# trials whose below-threshold MAE is within this fraction of the front's best MAE,
# pick the one with the highest exceedance F-beta.
PARETO_MAE_TOLERANCE = 0.10
