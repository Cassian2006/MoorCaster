from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics.forecast import _read_series, forecast_series


def _daily_series() -> pd.Series:
    idx = pd.date_range("2024-04-01", periods=5, freq="1D")
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)


def test_forecast_series_methods() -> None:
    s = _daily_series()

    naive = forecast_series(s, horizon=3, method="naive")
    ma = forecast_series(s, horizon=2, method="ma", ma_window=2)
    seasonal = forecast_series(s, horizon=4, method="seasonal", seasonal_period=2)
    seasonal_ma = forecast_series(s, horizon=2, method="seasonal_ma", ma_window=2, seasonal_period=2)

    assert naive.tolist() == [5.0, 5.0, 5.0]
    assert ma.tolist() == [4.5, 4.5]
    assert seasonal.tolist() == [4.0, 5.0, 4.0, 5.0]
    assert seasonal_ma.tolist() == [4.25, 4.75]


def test_read_series_resample_mean(tmp_path: Path) -> None:
    src = tmp_path / "curve.csv"
    df = pd.DataFrame(
        [
            {"time_bin": "2024-04-01 00:00:00", "idle_mmsi": 10},
            {"time_bin": "2024-04-01 12:00:00", "idle_mmsi": 20},
            {"time_bin": "2024-04-02 00:00:00", "idle_mmsi": 30},
        ]
    )
    df.to_csv(src, index=False)

    series = _read_series(
        input_csv=src,
        time_col="time_bin",
        value_col="idle_mmsi",
        target_freq="1D",
        agg="mean",
    )
    assert len(series) == 2
    assert round(float(series.iloc[0]), 2) == 15.00
    assert round(float(series.iloc[1]), 2) == 30.00
