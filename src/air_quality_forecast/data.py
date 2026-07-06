"""Load raw EPA AQS hourly PM2.5 files and reduce to a clean per-site hourly series."""
import pandas as pd

from . import config


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
