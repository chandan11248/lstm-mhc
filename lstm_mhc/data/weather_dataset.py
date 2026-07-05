"""NOAA Weather Dataset loader and preprocessor.

Source: JFK Airport hourly weather observations (2010-2018).
Kaggle: https://www.kaggle.com/datasets/zhaodianwen/noaaweatherdatajfkairport

Target variables (4 streams matching n=4):
    1. Temperature (HourlyDryBulbTempF)
    2. Humidity    (HourlyRelativeHumidity)
    3. Pressure    (HourlyStationPressure)
    4. Wind Speed  (HourlyWindSpeed)

Pipeline correctness (review §2 / C4)
-------------------------------------
- **No train/val/test leakage.** The scaler (mean/std) is fit on the training
  split only, and a configurable *guard gap* (default = max horizon) is left
  between consecutive splits so a window's targets never sit adjacent to a
  neighboring split's inputs.
- **Windowing respects splits.** Each split builds windows from its own slice
  after the gap is carved out, so a val/test input never reaches back into the
  training region.
- **Stride is configurable.** Stride-1 windows are heavily overlapping; this
  is documented and the autocorrelation can be reported. A non-overlapping
  protocol (``window_stride >= max_horizon``) is available for robustness.
- **Missingness is reported**, never silently imputed into the training signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Column names in the raw JFK CSV.
RAW_COLUMNS = [
    "DATE",
    "HOURLYDRYBULBTEMPF",
    "HOURLYRelativeHumidity",
    "HOURLYStationPressure",
    "HOURLYWindSpeed",
]
CLEAN_NAMES = {
    "HOURLYDRYBULBTEMPF": "temperature",
    "HOURLYRelativeHumidity": "humidity",
    "HOURLYStationPressure": "pressure",
    "HOURLYWindSpeed": "wind_speed",
}


def clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters (e.g. trace 'T' flags) and cast to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def load_and_clean_weather(csv_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load raw NOAA weather CSV, clean it, resample to hourly, and report missingness.

    Steps:
        1. Select the 4 target columns + DATE.
        2. Clean non-numeric characters.
        3. Parse DATE, set/sort index.
        4. Resample to a uniform 1h grid (mean within the hour).
        5. Report per-variable missingness *before* imputation.
        6. Linear-interpolate gaps (both directions), then ffill/bfill.
        7. Drop any residual NaNs.

    Args:
        csv_path: path to the raw ``jfk_weather_cleaned.csv``.
        columns: optional subset of clean column names to keep (default all 4).
    """
    print(f"Loading weather data from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=RAW_COLUMNS, low_memory=False)

    for col in RAW_COLUMNS[1:]:
        df[col] = clean_numeric(df[col])

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
    df = df.rename(columns=CLEAN_NAMES)

    # Uniform hourly grid; mean within each hour handles duplicate timestamps.
    df = df.resample("1h").mean()

    # Report missingness BEFORE imputation (critical for honest reporting).
    missing_pct = df.isnull().mean() * 100
    print("  Per-variable missingness (pre-imputation):")
    for col, pct in missing_pct.items():
        print(f"    {col:14s}: {pct:5.2f}%  ({df[col].isnull().sum()} rows)")
    longest_gap = _longest_gap_lengths(df)
    print(f"  Longest contiguous NaN gap: {longest_gap} hours")

    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill().dropna()

    if columns is not None:
        df = df[columns]

    print(f"  Cleaned: {len(df)} hourly obs | {df.index[0]} -> {df.index[-1]} | "
          f"cols={list(df.columns)} | remaining NaN={df.isnull().sum().sum()}")

    if len(df) < 50000:
        print("  WARNING: <50,000 rows — this looks like SYNTHETIC data. "
              "Synthetic results are NOT valid for paper submission.")
    else:
        print(f"  Provenance: REAL NOAA JFK ({len(df):,} observations).")
    return df


def _longest_gap_lengths(df: pd.DataFrame) -> int:
    """Length (hours) of the longest run of NaNs across *any* column."""
    isna = df.isnull().any(axis=1).astype(int).values
    if isna.sum() == 0:
        return 0
    best = cur = 0
    for v in isna:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


class WeatherTimeSeriesDataset(Dataset):
    """Sliding-window dataset for multivariate time-series forecasting.

    Each item is ``(x, y)`` where ``x`` is the input window ``(input_len, F)``
    and ``y`` stacks the future targets ``(num_horizons, F)``.

    Args:
        data: ``(N, F)`` normalized array (already split + scaled).
        input_len: input window length T.
        horizons: future offsets to predict (e.g. [6,12,24,48,72]).
        stride: step between consecutive window starts (1 = dense overlap).
    """

    def __init__(
        self,
        data: np.ndarray,
        input_len: int = 724,
        horizons: Optional[List[int]] = None,
        stride: int = 1,
    ):
        self.data = data
        self.input_len = input_len
        self.horizons = horizons or [6, 12, 24, 48, 72]
        self.stride = stride
        self.max_horizon = max(self.horizons)

        # A window starting at index i needs [i : i+input_len] as input and
        # indices i+input_len+h as targets. Last legal start is the one whose
        # furthest target is still in-bounds.
        self.valid_starts = list(range(
            0, len(data) - self.input_len - self.max_horizon, stride
        ))
        self.n_samples = len(self.valid_starts)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.valid_starts[idx]
        x = torch.tensor(self.data[i : i + self.input_len], dtype=torch.float32)
        base = i + self.input_len
        y = np.stack([self.data[base + h] for h in self.horizons])
        return x, torch.tensor(y, dtype=torch.float32)


def _make_scaler(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score mean/std. Caller fits on training data only and applies everywhere."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def build_weather_dataloaders(
    csv_path: str,
    config,
) -> Tuple:
    """Build train/val/test DataLoaders from NOAA weather CSV (no leakage).

    Split protocol (review §2 / C4):
      ``train : [0, a)``, ``gap``, ``val : [a+gap, b)``, ``gap``, ``test : [b+gap, N)``

    The scaler is fit on the training split only and applied to all splits.
    Each split's windows are built from its own (post-gap) slice, so no window
    input ever reaches into a neighboring split.

    Returns:
        ``(train_loader, val_loader, test_loader, scaler_info)`` where
        ``scaler_info`` carries the fit-on-train mean/std (needed to
        de-normalize predictions for interpretable metrics).
    """
    df = load_and_clean_weather(csv_path, columns=config.weather_columns)
    data = df.values.astype(np.float32)
    N = len(data)

    gap = config.split_gap
    train_end = int(N * config.train_ratio)
    val_end = train_end + gap + int(N * config.val_ratio)
    # Guard against degenerate splits.
    val_end = min(val_end, N - gap - config.input_len - max(config.horizons) - 1)

    # Guard: after the clamp, the test split must still have enough rows for
    # at least one full window. (Was previously silent — produced an empty
    # test_slice that the dataset class turned into zero windows.)
    min_test_len = config.input_len + max(config.horizons) + 1
    if N - (val_end + gap) < min_test_len:
        raise ValueError(
            f"Test split too small after weather split math: "
            f"val_end={val_end}, N={N}, gap={gap}, "
            f"required >= {min_test_len} rows after val_end+gap. "
            f"Reduce train/val_ratio, input_len, or horizons."
        )

    train_slice = data[:train_end]
    val_slice = data[train_end + gap : val_end]
    test_slice = data[val_end + gap :]

    # Fit scaler on TRAIN ONLY (no leakage).
    train_mean, train_std = _make_scaler(train_slice)
    train_norm = (train_slice - train_mean) / train_std
    val_norm = (val_slice - train_mean) / train_std
    test_norm = (test_slice - train_mean) / train_std

    train_ds = WeatherTimeSeriesDataset(train_norm, config.input_len, config.horizons, config.window_stride)
    val_ds = WeatherTimeSeriesDataset(val_norm, config.input_len, config.horizons, stride=1)
    test_ds = WeatherTimeSeriesDataset(test_norm, config.input_len, config.horizons, stride=1)

    print(f"\nLeakage-free splits (gap={gap}h):")
    print(f"  Train [{0}:{train_end}]               : {len(train_ds)} windows")
    print(f"  Val   [{train_end+gap}:{val_end}]      : {len(val_ds)} windows")
    print(f"  Test  [{val_end+gap}:{N}]              : {len(test_ds)} windows")

    use_cuda = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True,
        num_workers=config.num_workers, pin_memory=use_cuda,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False,
        num_workers=config.num_workers, pin_memory=use_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False,
        num_workers=config.num_workers, pin_memory=use_cuda,
    )

    scaler_info = {
        "mean": train_mean,
        "std": train_std,
        "columns": list(df.columns),
        "fit_on": "train",
        "split_gap": gap,
    }
    return train_loader, val_loader, test_loader, scaler_info
