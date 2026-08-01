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

# Open-Meteo historical weather API (no key required).
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TARGET_COL = "pm25"
