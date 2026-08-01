"""Load raw EPA AQS hourly PM2.5 files and reduce to a clean per-site hourly series."""
import pandas as pd

from . import config

WEATHER_COLS = ["precipitation", "relative_humidity_2m"]


def load_raw() -> pd.DataFrame:
    frames = [pd.read_parquet(config.RAW_DIR / f"pm25_state{config.STATE_CODE}_{y}.parquet") for y in config.YEARS]
    df = pd.concat(frames, ignore_index=True)
    df["site_id"] = df["County_Code"].astype(str).str.zfill(3) + "-" + df["Site_Num"].astype(str).str.zfill(4)
    return df[df["site_id"].isin(config.SITES)].copy()


def to_hourly_series(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (site_id, hour) on a strictly regular hourly grid.

    Uses GMT timestamps (unambiguous, no DST fold/gap issues) as the canonical clock.
    Multiple POCs (co-located monitors) at the same site/hour are averaged.
    """
    raw = raw.copy()
    raw["dt"] = pd.to_datetime(raw["Date_GMT"] + " " + raw["Time_GMT"])
    raw = raw.rename(columns={"Sample_Measurement": config.TARGET_COL})

    per_site_hour = (
        raw.groupby(["site_id", "dt"], as_index=False)[config.TARGET_COL].mean()
    )

    out = []
    for site_id, g in per_site_hour.groupby("site_id"):
        g = g.set_index("dt").sort_index()
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_index)
        g["site_id"] = site_id
        g.index.name = "dt"
        out.append(g)

    return pd.concat(out).reset_index()


def load_weather() -> pd.DataFrame:
    """Read and concatenate all per-site weather parquets saved by scripts/download_weather.py.

    Returns columns [site_id, dt, precipitation, relative_humidity_2m]. Raises FileNotFoundError
    with a clear message if no weather parquets are present yet.
    """
    paths = sorted(config.RAW_DIR.glob(f"weather_state{config.STATE_CODE}_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No weather parquets found in {config.RAW_DIR}. Run scripts/download_weather.py first."
        )
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def attach_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Left-merge weather (precipitation, relative_humidity_2m) onto an hourly PM2.5 frame.

    Uses (site_id, dt) as the join key so it preserves the regular hourly grid and every
    PM2.5 row produced by to_hourly_series, regardless of weather coverage. This is the single
    function callers (feature builder, inference) should use to attach weather, keeping
    train/serve parity. If no weather data has been downloaded yet, returns `hourly` unchanged
    (with NaN-filled weather columns) so callers that don't need weather are never broken.
    """
    try:
        weather = load_weather()
    except FileNotFoundError:
        out = hourly.copy()
        for col in WEATHER_COLS:
            # float("nan") keeps the column float64 (not object/pd.NA), so downstream
            # rolling-window feature code works unchanged on the all-missing case.
            out[col] = float("nan")
        return out

    return hourly.merge(weather, on=["site_id", "dt"], how="left")
