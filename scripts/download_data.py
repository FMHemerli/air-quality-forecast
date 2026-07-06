"""Download EPA AQS pre-generated hourly PM2.5 (parameter 88101) files and filter to a target state.

Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html (public domain, no API key).
Downloads the full national file per year, filters to STATE_CODE, and discards the
national CSV to keep local disk usage small.
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# California, to capture wildfire-smoke PM2.5 spikes alongside routine urban/valley pollution.
DEFAULT_STATE_CODE = "06"


def download_year(year: int, state_code: str = DEFAULT_STATE_CODE) -> Path:
    out_path = RAW_DIR / f"pm25_state{state_code}_{year}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path

    url = f"{BASE_URL}/hourly_88101_{year}.zip"
    print(f"[download] {url}")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        print(f"[extract] {csv_name} ({len(resp.content) / 1e6:.1f} MB zipped)")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    # EPA's schema alternates between space- and underscore-separated column names by year.
    df.columns = [c.replace(" ", "_") for c in df.columns]

    df = df[df["State_Code"].astype(str).str.zfill(2) == state_code].copy()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} ({len(df):,} rows, state={state_code})")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--state-code", default=DEFAULT_STATE_CODE)
    args = parser.parse_args()

    for year in args.years:
        download_year(year, args.state_code)
