# A 24-hour guideline inside an hourly forecast

A note on something that surfaced while building the exceedance detection, and that the
current implementation still partly gets wrong.

> **Status update.** The detection threshold has since moved from 15 to 35 µg/m³, which was
> the open question at the end of this note. That change is now implemented and measured;
> see [What moving to 35 actually bought](#what-moving-to-35-actually-bought) below. Of the
> three objections raised here, one (inflated base rate) is materially reduced and two
> (averaging-window mismatch, diurnal bias) are untouched. The note is kept in its original
> form, with the resolution appended, because the reasoning is what makes the result
> readable.

## The mismatch

The WHO guideline for PM2.5 — 15 µg/m³ — is a **24-hour mean**. This forecast is hourly,
and the exceedance metric compares hourly readings against that number as if it were an
hourly limit. Those are not the same quantity, and treating them as interchangeable is
convenient rather than correct.

It is easy to miss because the units match. Both are µg/m³, so the comparison type-checks.
What doesn't match is the averaging window, and that turns out to matter more than it
first appears.

## Why it matters

PM2.5 has a strong diurnal cycle, driven mostly by the atmospheric boundary layer. At
night the layer compresses and concentrations climb; by afternoon, vertical mixing dilutes
them. The same daily average can therefore hide very different hourly trajectories.

A single fixed threshold applied to every hour inherits that structure as bias. It
over-flags during the hours that are naturally dirty and under-flags during the hours that
are naturally clean — even across days whose 24-hour means are identical. The detector
ends up partly measuring what time it is rather than how bad the air is.

There is a second, subtler consequence. The event the guideline actually regulates is
"the daily mean exceeded 15". The label being optimized is "this hour exceeded 15". These
are different events with different base rates, and the model is being tuned for the one
that doesn't carry the health meaning.

The base rate is the third issue. Applied hour by hour in California, 15 µg/m³ is crossed
often. That makes exceedance a common event rather than a rare one, which inflates the
F-beta score by construction. The detection metric looks healthy without the hard problem
having been solved — the most dangerous kind of good number.

## Where this could go

The most appealing idea is also the simplest. Pollutant concentrations are roughly
lognormal, which favours a multiplicative decomposition:

$$x_t = m_d \cdot r_{h(t)} \cdot \eta_t$$

where $m_d$ is the day's mean, $r_h$ is a climatological diurnal profile (the typical
ratio of hour $h$ to the daily mean, averaging to 1), and $\eta$ is multiplicative noise.
Rearranged, this gives an hour-specific threshold:

$$\tau_h = 15 \cdot r_h$$

That is a static lookup table — twenty-four numbers per site, estimated once offline. It
corrects the diurnal bias directly and stays physically readable: during compressed
boundary-layer hours $r_h > 1$ and the bar rises; during well-mixed hours it drops. Since
the project already commits to asymmetric cost (recall weighted four times precision via
$\beta = 2$), the threshold can be shifted by a quantile of $\eta$ to buy recall at the
price of false alarms — one interpretable knob.

A more expressive alternative is to stop thresholding raw values and estimate the
conditional probability directly:

$$p_t = P\left(m_d > 15 \mid x_t,\ h,\ \text{site},\ \text{recent history}\right)$$

then calibrate it (Platt or isotonic) and check it honestly with a reliability diagram and
Brier score. This uses more than the hour of day, and the existing window features
(`roll_24h_mean`, `roll_24h_max`, the lags) already carry the recent history, so nothing
new is needed at inference time. The cost is that a probability is only useful if it is
genuinely calibrated, and that requires its own validation.

Underlying both is a question worth answering first, because it is cheap and it bounds
how well either approach can possibly work. Because hourly PM2.5 is strongly
autocorrelated, a 24-hour mean is not an average of 24 independent samples. The effective
sample size

$$n_{\text{eff}} = \frac{24}{1 + 2\sum_{k=1}^{23}\left(1 - \frac{k}{24}\right)\rho_k}$$

says how much smoothing actually happens. The smaller it is, the more a single hour tells
you about the day it belongs to — and the more plausible any hourly threshold becomes.

## What would make it convincing

The evaluation has to move to the level the guideline regulates. Recall, precision and
F-beta should be computed over **days**, asking whether the scheme recovers the days whose
true mean exceeded 15 — not over hours.

It also needs a ceiling. A detector allowed to see the entire day sets the upper bound on
achievable skill, and the gap to it is the part no hourly method can recover. One hour
cannot determine a 24-hour mean; that limit should be measured and reported rather than
quietly ignored. The existing ablation harness in `scripts/ablate.py` can carry most of
this comparison with little adaptation.

## Open questions

Which 24-hour window is the right one — the calendar day, as the guideline is usually
applied, or a rolling 24-hour mean? The choice changes the label and the base rate, so it
isn't a detail.

Does $r_h$ need to vary by season? Probably, in the Central Valley, where winter
temperature inversions reshape the daily cycle. Does it need to vary by site? Almost
certainly — Fresno and Monterey sit at opposite ends of the pollution regime range covered
here.

And is 15 µg/m³ even the interesting threshold for hourly work? The US AQI
"Unhealthy for Sensitive Groups" boundary at 35 µg/m³ produces rarer, sharper events, and
may be the more honest target for an hourly detector. Worth measuring both rather than
arguing about it.

## What moving to 35 actually bought

This last question has now been answered by measurement rather than argument. 35 µg/m³ is
the primary detection threshold; 15 is still reported alongside it as an evaluation-only
readout on the same frozen predictions, so the two are directly comparable.

**The base-rate objection is materially reduced.** Exceedance falls from ~13% of training
hours to ~2.0%. The detection score is no longer flattered by an event that happens all the
time — which is what the third objection above was about.

**The other two objections stand unchanged.** The US AQI band is *also* defined on a 24-hour
average (35.5–55.4 µg/m³ under the 2024 revision, of which 35.0 here is a round-number
stand-in), so the averaging-window mismatch is untouched. The diurnal-bias argument arguably
gets *sharper*, not weaker: a rarer threshold is crossed almost exclusively during
compressed-boundary-layer hours, so the detector's dependence on time-of-day increases. The
hour-specific threshold above simply becomes $\tau_h = 35 \cdot r_h$.

**A cost that had to be paid to measure this at all.** Validating on 2024-H1 gave only 65
hours above 35 out of 21,316 — 0.30%. An F-beta estimated over 65 positives is noise, and the
failure would have been silent, since `exceedance_scores` returns 0.0 rather than raising.
Validation was therefore extended backwards to 2023-07-01, picking up the 546 exceedances of
2023-H2 for 611 in total. Training data shrank 25%.

The test window was deliberately **not** moved. Extending it backwards over 2024-H1 would have
produced a larger test set, but 2024-H1 was the validation window under the previous split —
the feature set and search ranges this project inherited were chosen with it visible. Keeping
it in validation preserves both the honesty of "never seen during hyperparameter search" and
direct comparability with every number published before the change.

That comparability is worth what it cost. On the identical test window, at the identical
persistence baseline (RMSE 4.45 / 7.27 / 9.09, unchanged by construction), 25% less training
data cost +0.03 RMSE at 4h and −0.01 at 12h. The reduction was essentially free where it
matters.

**The result.** Detection F-beta (β=2) on the test split, model versus persistence baseline:

| horizon | F₂ @35 model | F₂ @35 persistence | F₂ @15 model | F₂ @15 persistence |
|---|---|---|---|---|
| 1h | 0.690 | **0.758** | 0.743 | **0.755** |
| 4h | **0.531** | 0.496 | **0.669** | 0.618 |
| 12h | **0.328** | 0.288 | **0.611** | 0.506 |

The absolute scores at 35 are much lower than at 15, exactly as predicted: rarer events are
harder, and the numbers at 15 were partly inflated by a base rate of ~13% of hours. What does
not change is the sign — the model beats persistence at 4h and 12h under both thresholds, and
loses at 1h under both. So the deflation is real, and it is uniform: it did not selectively
remove the model's advantage.

That last point is not something this note originally expected, and it was nearly recorded the
other way. An earlier version of this migration pooled 2024-H1 into the test set; there, the
4h comparison read as a narrow *loss* (0.458 vs 0.464), and it was written up as the model
having lost half its evidence against the clock objection. It had not. 2024-H1 carries a 0.30%
base rate, where both model and baseline score near-nothing, and pooling it dragged the
aggregate down. A metric averaged across two regimes that differ by 8x in base rate mostly
reports which regime dominates the sample.

## Open questions after this change

Moving the threshold did not touch the averaging window, so the daily-mean reformulation
described above is still the substantive next step, and the calendar-only ablation control is
still unrun. Both matter more now, not less: with only one horizon still beating persistence
on detection, there is less margin absorbing an unattributed clock effect.

## A consequence already visible in the ablation

This isn't only a future concern — it contaminates a result already measured. In the feature
ablation, the groups that raise exceedance recall (precipitation at 1h and 4h, wavelet at 12h)
are among those that best represent the diurnal and annual cycles. The FFT diurnal band
energy, the day-of-year harmonics, and a climatological anomaly indexed by (site, month, hour)
are all, among other things, high-resolution clocks.

If the exceedance label is partly a function of what hour it is, then a better clock buys
recall without buying any forecasting skill. The ablation as run cannot separate the two.

Two things argue against the strong version of that objection, and both survive the move to
35 µg/m³: the `base` combination already carries cyclical hour/day-of-week/month encodings, so
every reported delta is incremental over a clock-aware baseline; and the model beats the
persistence baseline on exceedance F-beta at 4h (0.531 vs 0.496) and 12h (0.328 vs 0.288),
which a pure clock could not do, since persistence carries information about the air and none
about the hour. At 15 µg/m³ the same two wins held (0.669 vs 0.618 and 0.611 vs 0.506), close
to what was published before the migration.

What would settle it is a **calendar-only detector** — hour, day-of-week, month and site, with
no pollution history whatsoever — added as another ablation combination. That control has
still not been run, and until it is, the 12h detection improvements should be read as
unattributed. It is cheap to add and remains the first thing to do when this study is picked
up — more so now, since a rarer threshold makes exceedance more concentrated in specific hours
of the day, not less.

One caution the reruns made concrete: ablation rankings are properties of a
(threshold, split) pair, not of the features. Reran at 35 against a test set that wrongly
absorbed 2024-H1, the 63-feature combination appeared to win RMSE at 4h, overturning the
published "wins RMSE at no horizon". Rerun against the correct test window, that claim holds
again — and the full set is now the *worst* combination at 1h on every metric. The apparent
reversal was the test window, not the features.

## Cost of acting on this

Changing the target definition touches `metrics.py`, the Optuna objective, and every
number currently published in `models/metrics.json`. It means retraining and rewriting the
results section. It does not affect the feature pipeline or the causality guarantees —
that part stands either way.

**Correction, written after actually doing it:** the last sentence was wrong. Moving the
threshold alone would not have touched the feature pipeline, but it could not be measured
without also moving the split — and `features.py` imports `splits.TRAIN_END` to fit the
(site, month, hour) climatology on pre-cutoff rows only. So `data/processed/features.parquet`
had to be rebuilt too, and skipping that rebuild would have leaked the new validation window
into `seas_climatology` and its two derived anomaly columns, silently, with no error and a
*better*-looking validation score. The causality guarantees held, but only because the
rebuild was run; the guarantee is conditional on a step this note originally did not know
about. Anything that changes `TRAIN_END` must re-run `scripts/build_features.py`.
