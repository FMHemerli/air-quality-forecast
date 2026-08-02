"""Time-based train/val/test split. Never split randomly on time-series data — that leaks
future information into training via adjacent-hour correlation.

Only TRAIN_END moved when the detection threshold went to 35 µg/m³; the test window is
deliberately unchanged. California PM2.5 is strongly seasonal and the two halves of the year
are not interchangeable: H1 is the clean half, H2 carries the wildfire season and the onset
of Central Valley winter inversions. Validating on 2024-H1 alone gave only 65 hours above 35
out of 21,316 (0.30%) — too few to estimate an F-beta from, so Pareto selection would have
picked trials by noise. Extending validation back to 2023-07-01 adds the 546 exceedances of
2023-H2, for 611 in total.

The test window was NOT extended backwards to absorb 2024-H1, even though that would have
been the larger test set. 2024-H1 was the validation window under the previous split, so the
feature set and hyperparameter search ranges inherited by this project were chosen with it
visible. Promoting it to test would have made "never seen during hyperparameter search" false
at the project level while remaining true of any single training run — the kind of claim that
is technically defensible and substantively wrong. It stays in validation, which is the role
it already had.

Note that `split` puts no upper bound on the test set: it is "VAL_END onward", which runs a
few hours into 2025, not the 2024-H2 calendar half exactly.

IMPORTANT: `features._add_seasonal_features` imports TRAIN_END to fit the (site, month,
hour) climatology on pre-cutoff rows only. Changing the constants below therefore
invalidates data/processed/features.parquet — re-run `scripts/build_features.py` before
training, or the seasonal features will carry statistics estimated over what is now the
validation window.
"""
import pandas as pd

TRAIN_END = pd.Timestamp("2023-07-01")
VAL_END = pd.Timestamp("2024-07-01")
# test: 2024-07-01 onward (unchanged across the move to a 35 µg/m³ threshold)


def split(df: pd.DataFrame, dt_col: str = "dt"):
    train = df[df[dt_col] < TRAIN_END]
    val = df[(df[dt_col] >= TRAIN_END) & (df[dt_col] < VAL_END)]
    test = df[df[dt_col] >= VAL_END]
    return train, val, test
