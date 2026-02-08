from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.forecast import _default_seasonal_period, _infer_freq, forecast_series


def _pick_time_col(df: pd.DataFrame) -> str:
    if "date" in df.columns:
        return "date"
    if "time_bin" in df.columns:
        return "time_bin"
    raise KeyError("Input must contain date or time_bin column")


def _forecast_one(
    series: pd.Series,
    horizon: int,
    method: str,
    ma_window: int,
    seasonal_period_override: int,
):
    series = series.dropna().astype(float)
    if series.empty:
        return None, None, None
    if len(series) < 2:
        freq = "1D"
    else:
        freq = _infer_freq(series.index)
    seasonal_period = seasonal_period_override or _default_seasonal_period(freq) or 0
    vals = forecast_series(
        series=series,
        horizon=horizon,
        method=method,
        ma_window=ma_window,
        seasonal_period=seasonal_period,
    )
    future_index = pd.date_range(
        start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )
    return future_index, vals, freq


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast daily waiting time metrics (mean/p90)")
    parser.add_argument("--input", default="outputs/metrics/waiting_time_by_day.csv")
    parser.add_argument("--output", default="outputs/metrics/waiting_forecast.csv")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--method", default="seasonal_ma", choices=["naive", "ma", "seasonal", "seasonal_ma"])
    parser.add_argument("--ma-window", type=int, default=7)
    parser.add_argument("--seasonal-period", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    time_col = _pick_time_col(df)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    idx_df = df.set_index(time_col)
    out = None
    used_freq = "1D"

    targets = [("mean", "mean_forecast"), ("p90", "p90_forecast")]
    for src_col, out_col in targets:
        if src_col not in idx_df.columns:
            continue
        future_index, vals, freq = _forecast_one(
            series=idx_df[src_col],
            horizon=args.horizon,
            method=args.method,
            ma_window=args.ma_window,
            seasonal_period_override=args.seasonal_period,
        )
        if future_index is None:
            continue
        used_freq = freq
        sub = pd.DataFrame({"date": future_index, out_col: vals.values})
        out = sub if out is None else out.merge(sub, on="date", how="outer")

    if out is None:
        out = pd.DataFrame(columns=["date", "mean_forecast", "p90_forecast", "method", "freq"])
    else:
        out = out.sort_values("date")
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["method"] = args.method
        out["freq"] = used_freq

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"waiting forecast -> {args.output}")


if __name__ == "__main__":
    main()

