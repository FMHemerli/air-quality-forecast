"""Download hourly weather (precipitation, relative humidity) from the Open-Meteo archive API.

Source: https://open-meteo.com/en/docs/historical-weather-api (public, no API key).
Site coordinates are taken from the EPA raw PM2.5 data (Latitude/Longitude columns) so the
weather series lines up with the exact monitor locations already used for training.
Saves one parquet per site, skip-if-exists, mirroring the style of download_data.py.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_quality_forecast import config, data

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def _site_coordinates(raw: pd.DataFrame) -> dict:
    """Map each site_id to its (latitude, longitude), taken from the first non-null row."""
    coords = {}
    for site_id, g in raw.groupby("site_id"):
        g = g.dropna(subset=["Latitude", "Longitude"])
        if g.empty:
            continue
        row = g.iloc[0]
        coords[site_id] = (float(row["Latitude"]), float(row["Longitude"]))
    return coords


def download_site_weather(
    site_id: str,
    latitude: float,
    longitude: float,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> Path:
    out_path = config.RAW_DIR / f"weather_state{config.STATE_CODE}_{site_id}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,relative_humidity_2m",
        "timezone": "GMT",
    }
    print(f"[download] {config.OPENMETEO_ARCHIVE_URL} site={site_id} lat={latitude} lon={longitude}")
    resp = requests.get(config.OPENMETEO_ARCHIVE_URL, params=params, timeout=180)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "site_id": site_id,
            "dt": pd.to_datetime(hourly["time"]),
            "precipitation": hourly["precipitation"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
        }
    )

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} ({len(df):,} rows, site={site_id})")
    return out_path


def main() -> None:
    raw = data.load_raw()
    coords = _site_coordinates(raw)

    for site_id in config.SITES:
        if site_id not in coords:
            print(f"[skip] {site_id} has no Latitude/Longitude in raw EPA data")
            continue
        latitude, longitude = coords[site_id]
        download_site_weather(site_id, latitude, longitude)


if __name__ == "__main__":
    main()
