# PM2.5 Forecast — California EPA AQS

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://fmhemerli-air-quality-forecast.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Optuna](https://img.shields.io/badge/Optuna-blueviolet)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Short-horizon (1h / 4h / 12h) PM2.5 particulate-matter forecasting for five California air
quality monitoring sites, built as an independent portfolio project on public data.

Built entirely from scratch on public EPA data — no code, data, configuration, or
methodology from any client engagement. See [Data & provenance](#data--provenance).

**Live demo**: https://fmhemerli-air-quality-forecast.streamlit.app/

## Problem

Predict PM2.5 concentration (µg/m³) 1, 4, and 12 hours ahead at five California EPA AQS
monitoring sites chosen for a mix of pollution regimes:

| Site | Region | Character |
|---|---|---|
| Fresno (019-5025) | Central Valley | high baseline, ag burning + wildfire smoke |
| Kings / Corcoran (031-0004) | Central Valley | high baseline, sharpest recorded spikes |
| Alameda / Oakland (001-0009) | SF Bay Area | cleaner urban baseline |
| Monterey / Salinas (053-1003) | Central Coast | low baseline, coastal |
| Ventura (111-1004) | South Coast | moderate urban/coastal mix |

## Approach

- **Data**: EPA Air Quality System (AQS) hourly PM2.5 mass concentration, 2022–2024, public
  domain, downloaded directly from `aqs.epa.gov` (no API key required).
- **Features**: lag features (1–48h), backward-looking rolling mean/std/max (3–24h windows,
  strictly shifted to avoid leakage), cyclical hour/day-of-week/month encodings. All feature
  logic lives in one shared module (`src/air_quality_forecast/features.py`) used identically
  by training and by the dashboard — no train/serve skew by construction.
- **Model**: one XGBoost regressor per horizon, hyperparameters tuned with Optuna (TPE
  sampler + median pruning), early-stopped against a time-ordered validation split.
- **Asymmetric weighting**: training samples above the 75th percentile of the training
  target distribution are up-weighted, since under-predicting a pollution spike is
  operationally worse than over-predicting a calm period. The threshold and weight are
  derived from the training data itself, not a fixed constant.
- **Evaluation**: strict time-based train/val/test split (train ≤ 2023, val = H1 2024,
  test = H2 2024 — never a random split, which would leak adjacent-hour correlation).
  Compared against a persistence baseline (predict the current reading holds).

## Results

Test split = H2 2024, never seen during training or hyperparameter search. Full numbers in
`models/metrics.json`.

| Horizon | Model RMSE | Persistence RMSE | Model MAE | Persistence MAE | Model vs. baseline |
|---|---|---|---|---|---|
| 1h | 5.29 | 4.45 | 3.11 | 2.55 | **worse** (+19% RMSE) |
| 4h | 6.26 | 7.27 | 3.76 | 4.22 | better (−14% RMSE) |
| 12h | 6.90 | 9.09 | 4.21 | 5.48 | better (−24% RMSE) |

(µg/m³, PM2.5)

**Honest limitation, not a bug**: at 1h ahead, the persistence baseline (assume the next
reading equals the current one) beats the tuned model. Hour-to-hour PM2.5 autocorrelation is
very strong, and the feature set here is purely historical (lags/rolling stats/calendar) —
no weather covariates (wind speed/direction, boundary-layer height), which is exactly the
signal that would let a model anticipate a change persistence can't see coming. At 4h and
12h, where persistence naturally decays, the model's use of longer-range temporal structure
and the asymmetric spike weighting pays off clearly. Reporting this instead of hiding it is
the point: never claim a win without a documented baseline, including when the baseline wins.

## Repo layout

```
scripts/download_data.py   download + filter EPA AQS hourly PM2.5 files
scripts/build_features.py  build the processed, model-ready feature table
scripts/train.py           Optuna-tuned XGBoost training per horizon
src/air_quality_forecast/  shared feature engineering, config, data loading (used by both
                            train.py and the dashboard)
dashboard/app.py           Streamlit app: actual vs backtested forecast, by site/horizon
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py       # ~3 years of CA hourly PM2.5, filtered from national files
python scripts/build_features.py      # builds data/processed/features.parquet
python scripts/train.py --trials 30   # trains models/model_{1,4,12}h.json + metrics.json

streamlit run dashboard/app.py
```

## Next steps — a problem worth thinking about

One thing this project currently gets wrong, stated plainly because it's more interesting
than pretending otherwise: the WHO PM2.5 guideline of 15 µg/m³ is a **24-hour mean**, and
the exceedance detection here compares **hourly** readings against it. The units match, so
the comparison looks reasonable. The averaging windows don't match, and that's the problem.

It matters because PM2.5 has a strong diurnal cycle driven by the boundary layer — night
compresses it, afternoon mixing dilutes it. A single fixed hourly threshold therefore
over-flags the naturally dirty hours and under-flags the naturally clean ones, partly
measuring what time it is instead of how bad the air is. It also means the label being
optimized ("this hour crossed 15") isn't the event the guideline regulates ("the day's mean
crossed 15"), and that hourly crossings are common enough in California to inflate the
detection score by construction.

The direction I find most promising is a multiplicative diurnal profile: since pollutant
concentrations are roughly lognormal, an hour-of-day threshold `τ_h = 15 · r_h` — a static
24-value lookup per site — corrects the bias while staying physically interpretable. The
alternative is to drop thresholding raw values entirely and estimate a calibrated
probability that the *day's* mean will exceed the guideline. Either way, the evaluation has
to move to the daily level, and needs a ceiling: one hour cannot determine a 24-hour mean,
and that limit should be measured rather than ignored.

Full reasoning, including the open questions, in
[`docs/24h-guideline-hourly-forecast.md`](docs/24h-guideline-hourly-forecast.md).

## Data & provenance

Data source: [EPA Air Quality System (AQS)](https://aqs.epa.gov/aqsweb/airdata/download_files.html),
U.S. government public domain data. This project, including all code, feature engineering,
modeling choices, and thresholds, was written independently against this public dataset.
It does not reuse any code, data, schema, naming, or calibrated constant from any private or
client codebase.

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [`LICENSE`](LICENSE).

Copyright (C) 2026 Flávio Manoel Santos Hemerli

You may use, modify and redistribute this code, including commercially, but any derivative work — including software you run as a networked service — must be released as open source under the same AGPL-3.0 terms.
