from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _infer_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "1D"
    freq = pd.infer_freq(index)
    if freq:
        return freq
    delta = (index[1] - index[0]).total_seconds()
    if delta <= 3600:
        return "1H"
    return "1D"


def _default_seasonal_period(freq: str) -> int:
    f = str(freq).upper()
    if "H" in f:
        return 24
    if "D" in f:
        return 7
    return 0


def _moving_average(values: pd.Series, window: int) -> float:
    window = max(int(window), 1)
    return float(values.tail(window).mean())


def forecast_series(
    series: pd.Series,
    horizon: int,
    method: str = "seasonal_ma",
    ma_window: int = 7,
    seasonal_period: int = 0,
) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty:
        return pd.Series([0.0] * horizon)
    horizon = max(int(horizon), 1)
    method = method.lower()

    if seasonal_period <= 0:
        seasonal_period = _default_seasonal_period(_infer_freq(s.index))

    vals = []
    if method == "naive":
        vals = [float(s.iloc[-1])] * horizon
    elif method == "ma":
        base = _moving_average(s, ma_window)
        vals = [base] * horizon
    elif method == "seasonal":
        if seasonal_period <= 0 or len(s) < seasonal_period:
            vals = [float(s.iloc[-1])] * horizon
        else:
            pattern = s.tail(seasonal_period).to_numpy()
            vals = [float(pattern[i % seasonal_period]) for i in range(horizon)]
    else:
        # seasonal_ma
        if seasonal_period > 0 and len(s) >= seasonal_period:
            pattern = s.tail(seasonal_period).to_numpy()
            ma = _moving_average(s, ma_window)
            vals = [float((pattern[i % seasonal_period] + ma) / 2.0) for i in range(horizon)]
        else:
            ma = _moving_average(s, ma_window)
            vals = [ma] * horizon
    return pd.Series(vals)


def _read_series(input_csv: Path, time_col: str, value_col: str, target_freq: str, agg: str) -> pd.Series:
    df = pd.read_csv(input_csv)
    if time_col not in df.columns:
        raise KeyError(f"missing column: {time_col}")
    if value_col not in df.columns:
        raise KeyError(f"missing column: {value_col}")
    ts = pd.to_datetime(df[time_col], errors="coerce")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(vals.values, index=ts).dropna()
    if target_freq:
        if agg == "sum":
            s = s.resample(target_freq).sum()
        elif agg == "max":
            s = s.resample(target_freq).max()
        else:
            s = s.resample(target_freq).mean()
    s = s.dropna()
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast congestion series")
    parser.add_argument("--input", default="outputs/metrics/congestion_curve.csv")
    parser.add_argument("--output", default="outputs/metrics/congestion_forecast.csv")
    parser.add_argument("--time-col", default="time_bin")
    parser.add_argument("--value-col", default="idle_mmsi")
    parser.add_argument("--target-freq", default="1D")
    parser.add_argument("--agg", default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--method", default="seasonal_ma", choices=["naive", "ma", "seasonal", "seasonal_ma"])
    parser.add_argument("--ma-window", type=int, default=7)
    parser.add_argument("--seasonal-period", type=int, default=0)
    args = parser.parse_args()

    input_csv = Path(args.input)
    s = _read_series(
        input_csv=input_csv,
        time_col=args.time_col,
        value_col=args.value_col,
        target_freq=args.target_freq,
        agg=args.agg,
    )
    fc = forecast_series(
        series=s,
        horizon=args.horizon,
        method=args.method,
        ma_window=args.ma_window,
        seasonal_period=args.seasonal_period,
    )

    freq = _infer_freq(s.index) if not s.empty else (args.target_freq or "1D")
    start = s.index[-1] + pd.tseries.frequencies.to_offset(freq) if not s.empty else pd.Timestamp.utcnow().floor("D")
    future_index = pd.date_range(start=start, periods=args.horizon, freq=freq)
    out = pd.DataFrame(
        {
            "time_bin": future_index.strftime("%Y-%m-%d"),
            "forecast_value": fc.values,
            "method": args.method,
            "freq": freq,
            "source_col": args.value_col,
        }
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"forecast -> {out_path}")


if __name__ == "__main__":
    main()

