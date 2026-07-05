"""ETT (Electricity Transformer Temperature) dataset loader.

Downloads from the public ETT repo on demand if the CSV is not already
present locally. Uses the identical split + normalization protocol as the
NOAA weather loader (review §2): gap-separated chronological splits,
scaler fit on train only, configurable window stride.

Reference: Zhou et al. "Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting" (AAAI 2021). Dataset: ETTh1, ETTh2 (hourly),
ETTm1, ETTm2 (15-min).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ETT_URLS = {
    "ETTh1.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

ETT_TARGET_COLS = ["OT", "HUFL", "HULL", "MULL"]


def _download_ett(csv_name: str, dest_dir: str) -> str:
    """Download an ETT CSV if not already present."""
    dest = Path(dest_dir) / csv_name
    if dest.exists():
        return str(dest)
    url = ETT_URLS.get(csv_name)
    if url is None:
        raise FileNotFoundError(f"Unknown ETT file: {csv_name}")
    try:
        import requests
        print(f"Downloading {csv_name} from {url}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        print(f"  Saved to {dest} ({dest.stat().st_size:,} bytes)")
        return str(dest)
    except ImportError:
        raise FileNotFoundError(
            f"Cannot download {csv_name}: `requests` not installed. "
            f"Download manually from {url} and place at {dest}."
        )


def load_ett(
    csv_path: Optional[str] = None,
    csv_name: str = "ETTh1.csv",
    columns: Optional[List[str]] = None,
    data_dir: str = "data/ett",
) -> pd.DataFrame:
    """Load and return an ETT DataFrame with the target columns.

    Args:
        csv_path: explicit path; if ``None`` the file is auto-downloaded.
        csv_name: file name (e.g. ``"ETTh1.csv"``).
        columns: columns to keep (default ``ETT_TARGET_COLS``).
        data_dir: where to cache downloaded CSVs.
    """
    if csv_path is None:
        csv_path = _download_ett(csv_name, data_dir)
    df = pd.read_csv(csv_path)
    # Parse the 'date' column.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    cols = columns or ETT_TARGET_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_name}: {missing}. Available: {list(df.columns)}")
    df = df[cols].copy()
    # NOTE: ETTm is 15-minute data. The previous implementation resampled it to
    # 1h here, which destroyed the granularity that makes ETTm vs ETTh a
    # meaningful comparison. Removed; ETTm is now kept at native 15-min
    # resolution. ETTh stays at 1h (its native cadence).
    # Impute.
    df = df.interpolate("linear", limit_direction="both").ffill().bfill().dropna()
    return df


def build_ett_dataloaders(
    config,
    csv_name: str = "ETTh1.csv",
    csv_path: Optional[str] = None,
) -> Tuple:
    """Build train/val/test DataLoaders for ETT (same protocol as weather)."""
    from .weather_dataset import WeatherTimeSeriesDataset, _make_scaler

    df = load_ett(csv_path, csv_name, data_dir="data/ett")
    data = df.values.astype(np.float32)
    N = len(data)

    gap = config.split_gap
    train_end = int(N * config.train_ratio)
    val_end = train_end + gap + int(N * config.val_ratio)
    val_end = min(val_end, N - gap - config.input_len - max(config.horizons) - 1)

    # Guard: after the clamp, the test split must still have enough rows for
    # at least one full window. (Was previously silent — produced an empty
    # test_slice that the dataset class turned into zero windows.)
    min_test_len = config.input_len + max(config.horizons) + 1
    if N - (val_end + gap) < min_test_len:
        raise ValueError(
            f"Test split too small after ETT split math: "
            f"val_end={val_end}, N={N}, gap={gap}, "
            f"required >= {min_test_len} rows after val_end+gap. "
            f"Reduce train/val_ratio, input_len, or horizons."
        )

    train_slice = data[:train_end]
    val_slice = data[train_end + gap : val_end]
    test_slice = data[val_end + gap :]

    train_mean, train_std = _make_scaler(train_slice)
    train_norm = (train_slice - train_mean) / train_std
    val_norm = (val_slice - train_mean) / train_std
    test_norm = (test_slice - train_mean) / train_std

    train_ds = WeatherTimeSeriesDataset(train_norm, config.input_len, config.horizons, config.window_stride)
    val_ds = WeatherTimeSeriesDataset(val_norm, config.input_len, config.horizons, stride=1)
    test_ds = WeatherTimeSeriesDataset(test_norm, config.input_len, config.horizons, stride=1)

    use_cuda = torch.cuda.is_available()
    kw = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=use_cuda)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, drop_last=False, **kw)
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, drop_last=False, **kw)

    scaler_info = {
        "mean": train_mean, "std": train_std, "columns": list(df.columns),
        "fit_on": "train", "split_gap": gap,
    }
    return train_loader, val_loader, test_loader, scaler_info
