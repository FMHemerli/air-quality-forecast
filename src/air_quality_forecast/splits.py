"""Time-based train/val/test split. Never split randomly on time-series data — that leaks
future information into training via adjacent-hour correlation."""
import pandas as pd

TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-07-01")
# test: 2024-07-01 onward


def split(df: pd.DataFrame, dt_col: str = "dt"):
    train = df[df[dt_col] < TRAIN_END]
    val = df[(df[dt_col] >= TRAIN_END) & (df[dt_col] < VAL_END)]
    test = df[df[dt_col] >= VAL_END]
    return train, val, test
