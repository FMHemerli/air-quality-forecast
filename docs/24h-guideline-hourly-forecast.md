# A 24-hour guideline inside an hourly forecast

A note on something that surfaced while building the exceedance detection, and that the
current implementation quietly gets wrong.

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

## Cost of acting on this

Changing the target definition touches `metrics.py`, the Optuna objective, and every
number currently published in `models/metrics.json`. It means retraining and rewriting the
results section. It does not affect the feature pipeline or the causality guarantees —
that part stands either way.
