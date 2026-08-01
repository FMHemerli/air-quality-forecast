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

# Health-exceedance threshold (µg/m3), WHO 24h PM2.5 guideline. Anchors both the
# training sample weighting (losses.sample_weights) and the multi-objective Optuna
# exceedance-detection metric (metrics.exceedance_scores), so train/serve parity holds
# on which threshold "counts" as a health exceedance. Configurable per deployment.
HEALTH_THRESHOLD_UGM3 = 15.0

# Beta for the F-beta exceedance-detection metric (metrics.exceedance_scores). beta > 1
# weights recall over precision, reflecting that missing a health exceedance is worse
# than a false alarm.
DETECTION_FBETA = 2.0

# Pareto-front selection tolerance (see scripts/train.py select_pareto_trial): among
# trials whose below-threshold MAE is within this fraction of the front's best MAE,
# pick the one with the highest exceedance F-beta.
PARETO_MAE_TOLERANCE = 0.10
