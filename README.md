# PM2.5 Forecast — California EPA AQS

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://fmhemerli-air-quality-forecast.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Optuna](https://img.shields.io/badge/Optuna-blueviolet)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Short-horizon (1h / 4h / 12h) PM2.5 particulate-matter forecasting for five California air
quality monitoring sites, built as an independent portfolio project.

Two public datasets: hourly PM2.5 from the EPA Air Quality System and hourly precipitation
and relative humidity from Open-Meteo ERA5 reanalysis, both fetched by scripts in this
repository. See [Data & provenance](#data--provenance).

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
  The detection threshold is 35 µg/m³, in the US AQI "Unhealthy for Sensitive Groups" region:
  below it, minimize regression error (MAE); above it, maximize exceedance detection (F-beta,
  β=2, weighting recall 4:1). These goals conflict genuinely — a single weighted sum was
  rejected because the two metrics are incommensurable (µg/m³ vs dimensionless score),
  embedding an arbitrary exchange rate. The final model is picked from the Pareto front by
  selecting the highest F-beta among trials whose below-threshold MAE is within 10% of the
  front's best. The WHO guideline (15 µg/m³) is still reported alongside, as an
  evaluation-only readout on the same frozen predictions.
  Detection and the asymmetric sample weighting are anchored on **different** constants
  (35 and 15): at an anchor of 35 only 2.0% of training rows get any upweight, at a mean
  weight of 1.014, which is near enough to an identity transform that the "missing a spike
  costs more than a false alarm" mechanism would stop doing anything.
- **Evaluation**: strict time-based train/val/test split (train 2022-01→2023-06, val =
  2023-07→2024-06, test = H2 2024 — never a random split, which would leak adjacent-hour
  correlation). Validation was extended backwards when the threshold moved to 35: validating
  on 2024-H1 alone gave only 65 hours above 35 out of 21,316, too few to estimate an F-beta
  from, while adding 2023-H2 brings the total to 611. The **test window was left untouched**,
  so every number here stays comparable to what was published before the change, and 2024-H1
  is never promoted from validation to test. Compared against a persistence baseline (predict
  the current reading holds).

## Results

Test split = H2 2024, never seen during training or hyperparameter search — and deliberately
left untouched by the move to a 35 µg/m³ threshold, so these numbers are directly comparable
to the ones published before it. Full numbers in `models/metrics.json`.

| Horizon | Model RMSE | Persistence RMSE | Model MAE | Persistence MAE | RMSE before the change |
|---|---|---|---|---|---|
| 1h | 5.80 | 4.45 | 3.38 | 2.55 | 5.37 |
| 4h | 6.54 | 7.27 | 3.96 | 4.22 | 6.51 |
| 12h | 7.26 | 9.09 | 4.47 | 5.48 | 7.27 |

(µg/m³, PM2.5)

The training set lost 25% of its rows when validation was extended backwards, and at 4h and
12h that cost essentially nothing (+0.03 and −0.01 RMSE). At 1h it cost 0.42 — at the one
horizon where the model already loses to persistence.

Detection performance, at both thresholds, on the same predictions:

| Horizon | Recall @35 | F₂ @35 | F₂ @35 persistence | F₂ @15 | F₂ @15 persistence |
|---|---|---|---|---|---|
| 1h | 0.708 | 0.690 | **0.758** | 0.743 | **0.755** |
| 4h | 0.524 | **0.531** | 0.496 | **0.669** | 0.618 |
| 12h | 0.308 | **0.328** | 0.288 | **0.611** | 0.506 |

The F-beta values at 35 are much lower than at 15, and that is expected rather than a
regression: rarer events are harder, and the scores at 15 were partly inflated by a base rate
of ~13% of hours versus ~2%. The two columns are not comparable to each other — that is
precisely why both are reported. What does carry across is the *sign*: the model beats
persistence at 4h and 12h under both thresholds, and loses at 1h under both.

**Honest limitation, not a bug**: at 1h ahead, the persistence baseline (assume the next
reading equals the current one) beats the tuned model (5.80 vs 4.45 RMSE). Hour-to-hour
PM2.5 autocorrelation is very strong, and adding precipitation and humidity did not change
that — the ablation shows no treatment closes the 1h gap. The covariates that would let a
model anticipate a change persistence can't see coming are the transport ones still missing
here: wind speed and direction, and boundary-layer height. At 4h and 12h, where persistence
naturally decays, the model's use of longer-range temporal structure pays off clearly: it
wins RMSE, MAE, and exceedance F-beta at both horizons, under both thresholds. Reporting the
1h result instead of hiding it is the point: never claim a win without a documented baseline,
including when the baseline wins.

### Ablation study

`scripts/ablate.py` trains each feature-group combination and compares them on the same
threshold-aware metrics. The honest finding: **there is no single winner — it depends on
the horizon and which metric you care about.**

Real deltas vs the `base` (26-feature) combination, measured at the 35 µg/m³ threshold:

- **1h**: FFT gives the best RMSE (5.38 vs base 5.41, −0.03), wavelet essentially ties it
  (−0.01) while giving the best detection (F-beta 0.655 vs 0.637, +0.019). The full set is
  worst on every column at once (RMSE +0.49, F-beta −0.060).
- **4h**: wavelet dominates error (RMSE 6.25 vs base 6.40, −0.16; below-threshold MAE −0.20),
  with FFT close behind (−0.12). Seasonal is the only group that buys detection
  (recall 0.496 → 0.538, F-beta +0.017) and pays the most for it (RMSE +0.50).
- **12h**: wavelet again wins error (RMSE 6.92 vs base 7.20, −0.27; below-threshold MAE
  −0.26). The full 63-feature set buys the largest detection gain in the whole study
  (recall 0.445 → 0.564, F-beta +0.090) at the largest error cost (RMSE +0.43).

Three conclusions: (a) there is no single winner — the best group changes with both horizon
and metric; (b) more features is not better: the 63-feature combination wins RMSE at no
horizon, and at 1h it is the worst combination on every metric; (c) at the longest horizon the
treatments buy detection at the cost of error — the full set trades +0.43 RMSE for +0.09
F-beta — which is exactly the trade-off the multi-objective setup was built to make visible.
Wavelet is the one group that is consistently good on error at the horizons that matter.

**What this ablation cannot tell you.** Because the exceedance label is an hourly crossing
of a 24-hour standard (see [Next steps](#next-steps--a-problem-worth-thinking-about)), the
detection score is partly a function of the diurnal cycle: some hours are simply more likely
to exceed. The groups that gain recall — wavelet at 1h, seasonal at 4h, the full set at 12h —
are also among the ones best equipped to represent that cycle, so part of their gain may be a
better clock rather than a better forecast. Moving to 35 µg/m³ arguably *sharpens* this concern
rather than relieving it: a rarer threshold is crossed almost exclusively during compressed-boundary-layer
hours, so the label's dependence on time-of-day goes up, not down.

Two things argue against the strong version of that objection, and both survive the move to
35: the `base` combination already contains cyclical hour/day/month encodings, so every
reported delta is incremental over a clock-aware baseline; and the model beats the persistence
baseline on exceedance F-beta at 4h (0.531 vs 0.496) and 12h (0.328 vs 0.288), which a pure
clock could not do, since persistence carries information about the air and none about the
hour. The control that would settle it — a calendar-only detector, using no pollution history
at all — is still not run. Until it is, read the 12h detection gains as unattributed.

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

**Rebuild note**: `build_features.py` must be re-run whenever `splits.TRAIN_END` changes.
The seasonal climatology (`seas_climatology` and its two anomaly columns) is fitted on
pre-cutoff rows only, so a stale `features.parquet` would carry statistics estimated over what
has become the validation window — silently, with no error, and with a *better*-looking
validation score.

**Training note**: the profile is tuned for reasonable wall-clock on CPU rather than
maximum accuracy (learning-rate floor and 600-tree cap, with Optuna trials run in parallel).
This is a portfolio project. Full training plus the full ablation takes ~14 minutes on a
6-core CPU. Parallel trials mean the Optuna seed no longer gives byte-identical
reproducibility.

## Next steps — a problem worth thinking about

One thing this project still gets wrong, stated plainly because it's more interesting
than pretending otherwise: the detection threshold of 35 µg/m³ is defined on a **24-hour
mean**, and the exceedance detection here compares **hourly** readings against it. The units
match, so the comparison looks reasonable. The averaging windows don't match, and that's the
problem. This was equally true of the WHO 15 µg/m³ guideline used before, and moving to 35
did not fix it — the US AQI band is a 24-hour standard too.

It matters because PM2.5 has a strong diurnal cycle driven by the boundary layer — night
compresses it, afternoon mixing dilutes it. A single fixed hourly threshold therefore
over-flags the naturally dirty hours and under-flags the naturally clean ones, partly
measuring what time it is instead of how bad the air is. It also means the label being
optimized ("this hour crossed 35") isn't the event the standard regulates ("the day's mean
crossed 35").

What moving to 35 *did* fix is the third objection: at 15 µg/m³, hourly crossings were common
enough in California (~13% of hours) to inflate the detection score by construction. At 35
they run ~2%. The cost of finding that out was a wider validation window and 25% of the
training rows — which turned out to be nearly free at 4h and 12h, and to leave the model's
advantage over persistence intact at both.

The direction I find most promising is a multiplicative diurnal profile: since pollutant
concentrations are roughly lognormal, an hour-of-day threshold `τ_h = 35 · r_h` — a static
24-value lookup per site — corrects the bias while staying physically interpretable. The
alternative is to drop thresholding raw values entirely and estimate a calibrated
probability that the *day's* mean will exceed the standard. Either way, the evaluation has
to move to the daily level, and needs a ceiling: one hour cannot determine a 24-hour mean,
and that limit should be measured rather than ignored.

Full reasoning, including the open questions, in
[`docs/24h-guideline-hourly-forecast.md`](docs/24h-guideline-hourly-forecast.md).

### A second mismatch: a noisy point forecast against a hard threshold

The problem above is about the label. There is a second one on the prediction side, found
while working out what the 5.0 µg/m³ detection limit costs.

Exceedance is declared when `pred > 35`, so one number serves both as the concentration that
matters for health and as the decision boundary on a noisy point forecast. Those are different
quantities. Because the model regresses toward the mean it undershoots real spikes by 9–21
µg/m³ depending on horizon, which makes "the forecast itself must reach 35" stricter than the
health question actually asks. Picking a decision threshold `c*` on the validation split
instead transfers to test at every horizon:

| horizon | c* | F₂ at `pred > 35` | F₂ at `pred > c*` | persistence |
|---|---|---|---|---|
| 1h | 28.5 | 0.659 | 0.720 | 0.758 |
| 4h | 26.0 | 0.504 | 0.611 | 0.496 |
| 12h | 27.0 | 0.439 | 0.599 | 0.288 |

Not implemented, for two reasons. It buys recall with precision — at 4h precision falls from
0.574 to 0.348, roughly two false alarms per real one — and while β=2 asks for exactly that
trade, how far to take it is an operational decision that belongs in front of whoever runs the
alarm rather than inside a tuned constant. And these numbers come from a model fitted on train
only, so that `c*` could be chosen on validation without contamination; the deployed models are
refit on train+val and start from a different place (12h F₂ 0.439 here against 0.328 published),
so the size of the gain would have to be reconfirmed inside the real pipeline.

Three other routes against the measurement floor were tested and abandoned:

- **Dropping sub-MDL rows from `below_threshold_mae`** makes it worse, not better (1h:
  3.165 → 3.376). Those rows are 28.7% of the calm-regime sample but only 24.0% of its error —
  they are easy, not noisy, so removing them raises the average.
- **Isotonic recalibration** (model on train, calibrator on validation) improves MAE and
  destroys detection: 4h F₂ falls 0.504 → 0.350, because the fitted map sends 35 → 32.0 and
  fewer forecasts cross the line. It barely moves the calm-regime bias it was meant to fix,
  since a monotone map on predicted value cannot separate "predicted 4, truth 1" from
  "predicted 4, truth 4".
- **`reg:gamma`**, which in principle matches the heteroscedasticity better, collapses
  detection outright (12h F₂ 0.439 → 0.032): the log link shrinks high predictions toward the
  conditional geometric mean and the model stops crossing 35. The measured heteroscedasticity
  is sub-linear anyway — residual spread grows 4.5× across a 35× range in level — so gamma
  over-corrects. It is also not directly applicable, since 5.22% of targets are ≤ 0.

All three fail the same way: optimizing error on the physical scale is blind to 35 being a
decision boundary.

One finding is recorded here because it was nearly written up backwards. The model looks like
it over-predicts badly in the calm regime, by +2.4 to +4.2 µg/m³ below the detection limit.
Most of that is a conditioning artifact, not a defect: selecting rows where the *measured*
value is low preferentially selects downward noise, so any unbiased predictor shows positive
bias there. The persistence baseline shows it too (+0.65 / +1.96 / +3.17). Only the 1h excess
over persistence is real, and it belongs to a different problem than the detection limit.

Aggregating to 24-hour means would dissolve all of this rather than work around it: noise
averages down, the negatives cancel by design, and daily means sit above the detection limit.
That it is also the fix for the averaging-window mismatch above is not a coincidence — both
problems come from reading an hourly instrument as though it answered a daily question.

### The caveat that outranks the others: the detection scores do not reproduce

Every improvement discussed above is smaller than the noise between two runs of this pipeline.

Re-running `train.py` unchanged — same code, same data, same split, same seed — reproduces the
regression metrics almost exactly and the detection metrics not at all:

| horizon | RMSE published | RMSE re-run | F₂ published | F₂ re-run |
|---|---|---|---|---|
| 4h | 6.54 | 6.534 | 0.531 | 0.498 |
| 12h | 7.26 | 7.325 | 0.328 | **0.481** |

RMSE lands within 0.07. F₂ at 12h moves by 0.153, which is half the size of the number itself.

The mechanism is already noted in `train.py`: `n_jobs=3` runs trials concurrently, so each TPE
suggestion depends on which earlier trials happen to have finished, and `TPESampler(seed=42)`
no longer pins the search. That comment calls the results "statistically equivalent across
runs". For RMSE the table above says it is right. For exceedance detection at a ~2% base rate
it is wrong: roughly 450 positives per horizon, split across a Pareto front chosen partly on
F-beta, is not enough to make the selected trial stable.

What that costs, stated plainly:

- **Any single-run comparison smaller than ~0.15 F₂ at 12h is unfalsifiable.** That includes
  the lead-time objective tested against the current one, which came out +0.034 at 4h and
  −0.068 at 12h — inconsistent in sign, and therefore indistinguishable from run noise.
- **It reaches results already published here.** The ablation section ranks feature
  combinations on detection deltas far smaller than 0.15. `docs/24h-guideline-hourly-forecast.md`
  already warns that ablation rankings are properties of a (threshold, split) pair rather than
  of the features; that warning is stronger than it was written — they are properties of the
  *run* as well.
- It does **not** undermine the RMSE/MAE results, the persistence comparison on regression
  error, or anything about the feature causality guarantees.

The fix is repetition, not modeling: run each configuration over several seeds and report a
median with a spread instead of a point. It is cheap to describe and costs n× the training
time, which is why it has not been done here. It is also a precondition for the rest of this
section — until the detection metric is stable enough to measure a change, choosing a decision
threshold, adopting a lead-time objective, or ranking feature groups by recall are all
decisions made on a number that will not survive being computed twice.

## Data & provenance

Two public datasets, each downloaded by a script in this repository, so the whole pipeline
reproduces from nothing but a clone:

- **PM2.5** — [EPA Air Quality System (AQS)](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
  hourly mass concentration (parameter 88101), 2022–2024, U.S. government public domain.
  `scripts/download_data.py` pulls the pre-generated national annual files (no API key),
  filters to California, and discards the national CSV.
- **Weather** — [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api),
  serving ERA5 reanalysis (Copernicus Climate Change Service). `scripts/download_weather.py`
  fetches hourly precipitation and relative humidity at each monitor's own latitude/longitude,
  taken from the EPA rows themselves so the two series line up on location as well as time.

### What the instrument actually reports

Every site runs a Met One beta-attenuation monitor (BAM-1020 on 127,573 of 129,151 rows,
BAM-1022 on the rest). That shows up in the data in three ways worth knowing before trusting
any value below about 5 µg/m³.

**Negative concentrations, 2.79% of readings, down to −5.1 µg/m³.** There is no negative mass.
A BAM derives concentration from the *difference* between two beta-attenuation counts taken
before and after drawing air through a filter tape; when the real loading is near zero, that
difference is dominated by the counting statistics of radioactive decay and by tape response
to humidity. The negative value is the error term of a difference estimator, not a
concentration. Three independent checks say noise rather than assume it: the reported method
detection limit is 5.0 µg/m³ on every row and 99.1% of the negatives are smaller than that in
magnitude; the negatives concentrate at the *cleanest* sites (Ventura, median 5.0, is 6.42%
negative — Fresno, median 10.0, is 1.42%), which is what noise around zero predicts and a
physical process would not; and the noise scale estimated separately from local curvature is
σ ≈ 1.46 µg/m³, putting the detection limit at 3.4σ, about where the usual ~3σ convention
puts it.

They are deliberately not clipped to zero. Truncating negative noise while keeping positive
noise biases every average upward, and the quantity the standard regulates is a 24-hour mean —
the negatives have to survive in order to cancel the positives. EPA publishes them for that
reason, and this project keeps them.

**32.2% of all readings fall below the 5.0 µg/m³ detection limit**, and so does 28.7% of the
calm-regime sample that `below_threshold_mae` is computed over. That is a ceiling on
achievable accuracy in the calm regime, not something modeling can recover. What follows from
it is in [Next steps](#next-steps--a-problem-worth-thinking-about).

**Reporting resolution is not uniform.** Four of the five sites report whole µg/m³; only
Fresno reports tenths, and it switched partway through 2022. Nothing in the pipeline depends
on this, but the same feature carries different quantization depending on the site.

The methods, and why each was chosen:

- **Gradient-boosted trees (XGBoost), one model per horizon.** The predictors are tabular,
  heterogeneous, and interact non-linearly; boosted trees handle that without scaling or
  imputation assumptions. Separate models per horizon because 1h is nearly persistence and
  12h is nearly climatology — one shared model would have to compromise between them.
- **Causal feature construction.** Every lag, rolling, spectral, wavelet and denoising
  feature reads only from t−1 backwards, and the seasonal climatology is estimated solely
  from pre-cutoff rows. Verified by recomputing on truncated series (max absolute
  difference 0.0), because a leak here would inflate every downstream number silently.
- **Multi-objective Optuna (TPE) instead of a weighted sum.** Below-threshold MAE (µg/m³)
  and exceedance F-beta (dimensionless) are incommensurable; adding them would hide an
  arbitrary exchange rate inside a hyperparameter. The Pareto front keeps the trade-off
  explicit and the selection rule visible.
- **Time-based split, never random.** Adjacent hours are strongly correlated, so a random
  split leaks the future into training and reports a score that cannot occur in deployment.
- **Persistence baseline on every metric.** "The current reading holds" is the forecast a
  useful model has to beat; where it doesn't — 1h here — the README says so.

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [`LICENSE`](LICENSE).

Copyright (C) 2026 Flávio Manoel Santos Hemerli

You may use, modify and redistribute this code, including commercially, but any derivative work — including software you run as a networked service — must be released as open source under the same AGPL-3.0 terms.
