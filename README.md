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
- **Features**: 63 selectable columns in six groups, all strictly backward-looking and
  shared between training and dashboard (`src/air_quality_forecast/features.py`):
  * `base` (26): lag features (1–48h), backward-looking rolling mean/std/max (3–24h windows),
    cyclical hour/day-of-week/month encodings.
  * `precip` (13): hourly precipitation and relative humidity from Open-Meteo ERA5 reanalysis,
    aggregated as trailing rolling sums/means (3–48h windows). Rationale: wet deposition (rain
    scavenges particles) and hygroscopic growth (humidity raises measured mass).
  * `fft` (6): band energies (low / diurnal / high), dominant frequency, spectral entropy,
    computed over a trailing causal 48h window.
  * `wavelet` (4): energy per Daubechies-4 decomposition level (approximation + 3 detail
    levels), same trailing window. Wavelets localize transients better than Fourier.
  * `denoise` (4): causally wavelet-denoised PM2.5 (VisuShrink soft-thresholding, MAD noise
    estimate), keeping only the last reconstructed sample to stay strictly causal.
  * `seasonal` (10): day-of-year harmonics (k=1..3) plus climatological anomaly (deviation
    from the site/month/hour mean, estimated only from pre-train-cutoff rows — no leakage).
  Every spectral/wavelet/denoise feature uses a trailing causal window ending at t−1,
  verified by recomputing on truncated series (max absolute difference 0.0).
- **Model**: one XGBoost regressor per horizon, hyperparameters tuned with multi-objective
  Optuna (TPE sampler; no pruner — median pruning has no meaning without a single scalar
  objective), early-stopped against a time-ordered validation split.
  The WHO PM2.5 guideline (15 µg/m³) is the health threshold: below it, minimize regression
  error (MAE); above it, maximize exceedance detection (F-beta, β=2, weighting recall 4:1).
  These goals conflict genuinely — a single weighted sum was rejected because the two metrics
  are incommensurable (µg/m³ vs dimensionless score), embedding an arbitrary exchange rate.
  The final model is picked from the Pareto front by selecting the highest F-beta among
  trials whose below-threshold MAE is within 10% of the front's best.
- **Evaluation**: strict time-based train/val/test split (train ≤ 2023, val = H1 2024,
  test = H2 2024 — never a random split, which would leak adjacent-hour correlation).
  Compared against a persistence baseline (predict the current reading holds).

## Results

Test split = H2 2024, never seen during training or hyperparameter search. Full numbers in
`models/metrics.json`.

| Horizon | Model RMSE | Persistence RMSE | Model MAE | Persistence MAE |
|---|---|---|---|---|
| 1h | 5.37 | 4.45 | 3.15 | 2.55 |
| 4h | 6.51 | 7.27 | 3.98 | 4.22 |
| 12h | 7.27 | 9.09 | 4.48 | 5.48 |

(µg/m³, PM2.5)

**Honest limitation, not a bug**: at 1h ahead, the persistence baseline (assume the next
reading equals the current one) beats the tuned model (5.37 vs 4.45 RMSE). Hour-to-hour
PM2.5 autocorrelation is very strong, and adding precipitation and humidity did not change
that — the ablation shows no treatment closes the 1h gap. The covariates that would let a
model anticipate a change persistence can't see coming are the transport ones still missing
here: wind speed and direction, and boundary-layer height. At 4h and 12h, where persistence
naturally decays, the model's use of
longer-range temporal structure and the multi-objective setup pays off clearly. Reporting
this instead of hiding it is the point: never claim a win without a documented baseline,
including when the baseline wins.

### Ablation study

`scripts/ablate.py` trains each feature-group combination and compares them on the same
threshold-aware metrics. The honest finding: **there is no single winner — it depends on
the horizon and which metric you care about.**

Real deltas vs the `base` (26-feature) combination:

- **1h**: wavelet gives the best RMSE (5.31 vs base 5.60, −0.30); seasonal also helps
  (−0.24); FFT actually hurts (+0.13). Full set gives best below-threshold MAE (2.57 vs
  2.72) and best precision.
- **4h**: FFT dominates (RMSE 6.22 vs base 6.80, −0.58; below-threshold MAE −0.25);
  denoising also helps (−0.52). Precipitation improves RMSE but worsens below-threshold
  MAE (+0.12) while raising recall (0.680 to 0.713).
- **12h**: nearly every treatment worsens error but improves detection. Wavelet: RMSE
  roughly unchanged but recall rises 0.622 to 0.685 and F-beta 0.608 to 0.645.
  Precipitation: below-threshold MAE +0.33, recall 0.622 to 0.672.

Two conclusions: (a) more features is not better — the 63-feature combination wins RMSE
at no horizon; (b) at the longest horizon the treatments buy detection at the cost of
error, which is exactly the trade-off the multi-objective setup was built to make visible.

## Repo layout

```
scripts/download_data.py   download + filter EPA AQS hourly PM2.5 files
scripts/download_weather.py download Open-Meteo ERA5 hourly precipitation + humidity
scripts/build_features.py  build the processed, model-ready feature table
scripts/train.py           Optuna-tuned XGBoost training per horizon
scripts/ablate.py          train each feature-group combination, write ablation_report
src/air_quality_forecast/  shared feature engineering, config, data loading (used by both
                            train.py and the dashboard)
dashboard/app.py           Streamlit app: actual vs backtested forecast, by site/horizon
docs/                      design notes (24h-guideline mismatch, etc.)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py          # ~3 years of CA hourly PM2.5, filtered from national files
python scripts/download_weather.py       # ERA5 precipitation + humidity, required before features
python scripts/build_features.py         # builds data/processed/features.parquet
python scripts/train.py --trials 25      # trains models/model_{1,4,12}h.ubj + metrics.json
python scripts/ablate.py                 # optional: writes data/processed/ablation_report.{json,csv}

streamlit run dashboard/app.py
```

**Training note**: the profile is tuned for reasonable wall-clock on CPU rather than
maximum accuracy (learning-rate floor and 600-tree cap, with Optuna trials run in parallel).
This is a portfolio project. Full training plus the full ablation takes ~14 minutes on a
6-core CPU. Parallel trials mean the Optuna seed no longer gives byte-identical
reproducibility.

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
