"""Build the model-ready processed dataset from raw EPA AQS files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_quality_forecast import config, data, features  # noqa: E402


def main() -> None:
    raw = data.load_raw()
    hourly = data.to_hourly_series(raw)
    hourly = data.attach_weather(hourly)
    print(f"[hourly] {len(hourly):,} site-hours across {hourly['site_id'].nunique()} sites")
    print(hourly.groupby("site_id")[config.TARGET_COL].apply(lambda s: s.notna().mean()).rename("coverage"))

    feat = features.build_features(hourly)
    feat = features.add_targets(feat)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "features.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} ({len(feat):,} rows, {len(features.feature_columns(feat))} feature columns)")


if __name__ == "__main__":
    main()
