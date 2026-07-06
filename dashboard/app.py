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
    feat = features.build_features(hourly)
    return features.add_targets(feat)


@st.cache_resource
def load_model(horizon: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(tree_method="hist", enable_categorical=True)
    model.load_model(config.MODELS_DIR / f"model_{horizon}h.json")
    return model


@st.cache_data
def load_metrics() -> dict:
    with open(config.MODELS_DIR / "metrics.json") as f:
        return json.load(f)


def main() -> None:
    st.title("PM2.5 Forecast — California EPA AQS Monitoring Sites")
    st.caption(
        "Portfolio project. Data: EPA Air Quality System (AQS) hourly PM2.5, public domain. "
        "Not affiliated with, and does not use any code, data, or methods from, any client project."
    )

    df = load_featurized()
    metrics = load_metrics()

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
    predictable = view.dropna(subset=feat_cols, how="any")
    preds = model.predict(predictable[feat_cols]) if len(predictable) else []

    m = metrics[f"{horizon}h"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model RMSE (test)", f"{m['model']['rmse']:.2f} µg/m³")
    col2.metric(
        "Persistence baseline RMSE",
        f"{m['baseline_persistence']['rmse']:.2f} µg/m³",
        delta=f"{m['model']['rmse'] - m['baseline_persistence']['rmse']:.2f}",
        delta_color="inverse",
    )
    col3.metric("Model MAE (test)", f"{m['model']['mae']:.2f} µg/m³")
    col4.metric("Spike threshold (p75, train)", f"{m['spike_threshold_ugm3']:.1f} µg/m³")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["dt"], y=view[config.TARGET_COL], name="Actual PM2.5", line=dict(color="#2c3e50")))
    if len(predictable):
        fig.add_trace(go.Scatter(
            x=predictable["dt"] + pd.Timedelta(hours=horizon), y=preds,
            name=f"Forecast (+{horizon}h, made at source time)", line=dict(color="#e67e22", dash="dot"),
        ))
    fig.update_layout(
        title=f"{config.SITES[site_id]} — actual vs {horizon}h-ahead forecast",
        xaxis_title="Time (UTC)", yaxis_title="PM2.5 (µg/m³)", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("All-horizon metrics"):
        rows = []
        for h_key, hm in metrics.items():
            rows.append({
                "horizon": h_key,
                "model_rmse": hm["model"]["rmse"],
                "persistence_rmse": hm["baseline_persistence"]["rmse"],
                "model_mae": hm["model"]["mae"],
                "persistence_mae": hm["baseline_persistence"]["mae"],
                "n_test": hm["n_test"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
