"""Streamlit dashboard: PM2.5 forecasts for a handful of California monitoring sites.

Recomputes features from the raw hourly series using the exact same
`air_quality_forecast.features.build_features` function used at training time, then loads
the trained XGBoost models to predict — the same code path serves both training and this
"live" view, so there is no train/serve skew by construction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import xgboost as xgb  # noqa: E402

from air_quality_forecast import config, data, features  # noqa: E402

st.set_page_config(page_title="PM2.5 Forecast", layout="wide")


@st.cache_data(show_spinner="Loading and featurizing raw EPA AQS data...")
def load_featurized() -> pd.DataFrame:
    raw = data.load_raw()
    hourly = data.to_hourly_series(raw)
    hourly = data.attach_weather(hourly)
    feat = features.build_features(hourly)
    return features.add_targets(feat)


@st.cache_resource
def load_model(horizon: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(tree_method="hist", enable_categorical=True)
    model.load_model(config.MODELS_DIR / f"model_{horizon}h.ubj")
    return model


@st.cache_data
def load_metrics() -> dict:
    with open(config.MODELS_DIR / "metrics.json") as f:
        return json.load(f)


def validate_metrics(metrics: dict) -> list[str]:
    """Check that `metrics` (as loaded from models/metrics.json) has the shape this
    dashboard needs for every horizon in `config.FORECAST_HORIZONS_H`.

    Returns a list of human-readable descriptions of missing/malformed keys, empty if the
    structure is valid. This only covers the "core" fields the dashboard treats as required
    (model/baseline RMSE+MAE, the health threshold, and the presence of a threshold report);
    the exceedance sub-fields and the all-horizon expander table are additionally accessed
    with `.get()` fallbacks at display time, so a partially-populated file degrades to
    hiding those specific widgets instead of failing validation outright.
    """
    required_stat_keys = ("rmse", "mae")
    problems: list[str] = []
    for h in config.FORECAST_HORIZONS_H:
        h_key = f"{h}h"
        hm = metrics.get(h_key)
        if not isinstance(hm, dict):
            problems.append(f"'{h_key}' (missing horizon entry)")
            continue

        model = hm.get("model")
        if not isinstance(model, dict) or not all(k in model for k in required_stat_keys):
            problems.append(f"'{h_key}.model.{{rmse,mae}}'")

        baseline = hm.get("baseline_persistence")
        if not isinstance(baseline, dict) or not all(k in baseline for k in required_stat_keys):
            problems.append(f"'{h_key}.baseline_persistence.{{rmse,mae}}'")

        if "health_threshold_ugm3" not in hm:
            problems.append(f"'{h_key}.health_threshold_ugm3'")

        report = hm.get("model_threshold_report")
        if not isinstance(report, dict) or "below_threshold_mae" not in report or "exceedance" not in report:
            problems.append(f"'{h_key}.model_threshold_report.{{below_threshold_mae,exceedance}}'")

    return problems


def main() -> None:
    st.title("PM2.5 Forecast — California EPA AQS Monitoring Sites")
    st.caption(
        "Portfolio project. Data: EPA Air Quality System (AQS) hourly PM2.5, public domain. "
        "Not affiliated with, and does not use any code, data, or methods from, any client project."
    )

    df = load_featurized()
    metrics = load_metrics()

    missing = validate_metrics(metrics)
    if missing:
        st.error(
            "models/metrics.json is missing expected data: " + "; ".join(missing) + ". "
            "Regenerate it by running `python scripts/train.py`."
        )
        st.stop()

    site_id = st.sidebar.selectbox(
        "Site", options=list(config.SITES.keys()), format_func=lambda s: f"{config.SITES[s]} ({s})"
    )
    horizon = st.sidebar.selectbox("Forecast horizon (hours ahead)", options=config.FORECAST_HORIZONS_H, index=1)

    site_df = df[df["site_id"] == site_id].copy()
    min_dt, max_dt = site_df["dt"].min(), site_df["dt"].max()
    date_range = st.sidebar.slider(
        "Date range", min_value=min_dt.to_pydatetime(), max_value=max_dt.to_pydatetime(),
        value=(max_dt.to_pydatetime() - pd.Timedelta(days=30), max_dt.to_pydatetime()),
    )
    view = site_df[(site_df["dt"] >= date_range[0]) & (site_df["dt"] <= date_range[1])]

    feat_cols = features.feature_columns(df)
    model = load_model(horizon)

    target_col = f"target_{horizon}h"
    # XGBoost handles NaN features natively, so requiring every one of the 63 feature
    # columns to be non-null (as the old code did) drops far more rows than necessary —
    # spectral/climatology columns are legitimately NaN in early warm-up rows or sparse
    # climatology cells. Only require the core short-lag columns (see
    # features.core_lag_columns), which is what a row needs to be meaningfully predictable
    # at all.
    core_cols = features.core_lag_columns()
    predictable = view.dropna(subset=core_cols, how="any")
    preds = model.predict(predictable[feat_cols]) if len(predictable) else []

    m = metrics[f"{horizon}h"]
    # health_threshold_ugm3 and the exceedance sub-fields are validated as present for
    # every configured horizon by validate_metrics(), but the exceedance metric itself
    # (recall/fbeta) is not, so it's still accessed defensively here.
    threshold = m.get("health_threshold_ugm3")
    exceedance = (m.get("model_threshold_report") or {}).get("exceedance") or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model RMSE (test)", f"{m['model']['rmse']:.2f} µg/m³")
    col2.metric(
        "Persistence baseline RMSE",
        f"{m['baseline_persistence']['rmse']:.2f} µg/m³",
        delta=f"{m['model']['rmse'] - m['baseline_persistence']['rmse']:.2f}",
        delta_color="inverse",
    )
    col3.metric("Model MAE (test)", f"{m['model']['mae']:.2f} µg/m³")
    if "recall" in exceedance and "fbeta" in exceedance:
        col4.metric(
            "Exceedance recall (test)",
            f"{exceedance['recall']:.0%}",
            delta=f"F-beta {exceedance['fbeta']:.2f}",
            delta_color="off",
        )
    else:
        col4.metric("Exceedance recall (test)", "n/a")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["dt"], y=view[config.TARGET_COL], name="Actual PM2.5", line=dict(color="#2c3e50")))
    if threshold is not None:
        exceeded = view[view[config.TARGET_COL] > threshold]
        if len(exceeded):
            fig.add_trace(go.Scatter(
                x=exceeded["dt"], y=exceeded[config.TARGET_COL], mode="markers",
                name="Actual exceedance", marker=dict(color="#c0392b", size=5),
            ))
    if len(predictable):
        fig.add_trace(go.Scatter(
            x=predictable["dt"] + pd.Timedelta(hours=horizon), y=preds,
            name=f"Forecast (+{horizon}h, made at source time)", line=dict(color="#e67e22", dash="dot"),
        ))
    if threshold is not None:
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="#c0392b",
            annotation_text=f"WHO 24h guideline ({threshold:.0f} µg/m³)", annotation_position="top left",
        )
    fig.update_layout(
        title=f"{config.SITES[site_id]} — actual vs {horizon}h-ahead forecast",
        xaxis_title="Time (UTC)", yaxis_title="PM2.5 (µg/m³)", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The WHO guideline of 15 µg/m³ is defined as a 24-hour mean; the line above compares it "
        "directly against hourly readings, which is a convenient simplification, not a rigorous "
        "match (see docs/24h-guideline-hourly-forecast.md)."
    )

    with st.expander("All-horizon metrics"):
        rows = []
        for h_key, hm in metrics.items():
            model_stats = hm.get("model") or {}
            baseline_stats = hm.get("baseline_persistence") or {}
            threshold_report = hm.get("model_threshold_report") or {}
            exc = threshold_report.get("exceedance") or {}
            rows.append({
                "horizon": h_key,
                "model_rmse": model_stats.get("rmse"),
                "persistence_rmse": baseline_stats.get("rmse"),
                "model_mae": model_stats.get("mae"),
                "persistence_mae": baseline_stats.get("mae"),
                "below_threshold_mae": threshold_report.get("below_threshold_mae"),
                "exceedance_recall": exc.get("recall"),
                "exceedance_precision": exc.get("precision"),
                "exceedance_fbeta": exc.get("fbeta"),
                "n_test": hm.get("n_test"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
