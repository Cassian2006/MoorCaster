import argparse
from pathlib import Path
import sys

import pandas as pd

# Ensure project root is importable when running `python scripts/...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.forecast import _default_seasonal_period, _infer_freq, forecast_series


def _resample_day(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    fill_mode: str = "none",
    fill_value: float = 0.0,
) -> pd.Series:
    out = (
        df[[time_col, value_col]]
        .dropna()
        .assign(**{time_col: lambda x: pd.to_datetime(x[time_col])})
        .set_index(time_col)[value_col]
        .astype(float)
        .resample("1D")
        .mean()
    )
    if fill_mode == "none":
        out = out.dropna()
    elif fill_mode == "zero":
        out = out.fillna(float(fill_value))
    elif fill_mode == "ffill":
        out = out.ffill().fillna(float(fill_value))
    elif fill_mode == "interpolate":
        out = out.interpolate(method="time").ffill().bfill().fillna(float(fill_value))
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")
    return out


def _forecast(series: pd.Series, horizon: int, method: str, ma_window: int, seasonal_period: int) -> tuple[pd.Series, str]:
    if series.empty:
        raise ValueError("Series is empty, cannot forecast")
    try:
        freq = _infer_freq(series.index)
    except Exception:
        freq = "1D"
    seasonal = seasonal_period or _default_seasonal_period(freq) or 0
    vals = forecast_series(
        series=series,
        horizon=horizon,
        method=method,
        ma_window=ma_window,
        seasonal_period=seasonal,
    )
    future_index = pd.date_range(
        start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )
    return pd.Series(vals.values, index=future_index), freq


def _calc_scale(ais_daily: pd.Series, yolo_daily: pd.Series) -> tuple[float, int]:
    joined = pd.concat([ais_daily.rename("ais"), yolo_daily.rename("yolo")], axis=1).dropna()
    joined = joined[joined["yolo"] > 0]
    if joined.empty:
        return 1.0, 0
    ratios = joined["ais"] / joined["yolo"]
    scale = float(ratios.median())
    if scale <= 0:
        return 1.0, int(len(joined))
    return scale, int(len(joined))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S1+YOLO driven congestion forecast")
    parser.add_argument("--yolo-input", default="outputs/metrics/yolo_observed.csv")
    parser.add_argument("--ais-input", default="outputs/metrics/congestion_curve.csv")
    parser.add_argument("--ais-forecast-input", default="outputs/metrics/congestion_forecast.csv")
    parser.add_argument("--output", default="outputs/metrics/vision_forecast.csv")
    parser.add_argument("--series", default="idle_mmsi", choices=["idle_mmsi", "presence_mmsi"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--method", default="seasonal_ma", choices=["naive", "ma", "seasonal", "seasonal_ma"])
    parser.add_argument("--ma-window", type=int, default=7)
    parser.add_argument("--seasonal-period", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend weight for AIS forecast in [0,1]")
    parser.add_argument(
        "--yolo-fill-mode",
        default="interpolate",
        choices=["none", "zero", "ffill", "interpolate"],
        help="How to fill missing daily YOLO bins before forecasting",
    )
    parser.add_argument("--yolo-fill", type=float, default=0.0, help="Fallback value when fill requires constant")
    parser.add_argument(
        "--allow-missing-ais",
        action="store_true",
        help="Allow YOLO-only forecast when AIS inputs are unavailable",
    )
    parser.add_argument(
        "--default-scale-factor",
        type=float,
        default=1.0,
        help="Fallback scale factor when AIS is unavailable or unusable",
    )
    parser.add_argument(
        "--min-observed-points-for-seasonal",
        type=int,
        default=21,
        help="When AIS is missing and observed YOLO points are fewer than this, avoid seasonal methods.",
    )
    args = parser.parse_args()

    yolo_path = Path(args.yolo_input)
    ais_path = Path(args.ais_input)
    ais_forecast_path = Path(args.ais_forecast_input)

    if not yolo_path.exists():
        raise FileNotFoundError(f"Missing yolo input: {yolo_path}")

    yolo_df = pd.read_csv(yolo_path)
    if "time_bin" not in yolo_df.columns or "yolo_detections" not in yolo_df.columns:
        raise KeyError("yolo input must contain: time_bin, yolo_detections")
    yolo_observed_points = int(len(yolo_df))
    yolo_daily = _resample_day(
        yolo_df,
        "time_bin",
        "yolo_detections",
        fill_mode=args.yolo_fill_mode,
        fill_value=args.yolo_fill,
    )

    has_ais_series = False
    ais_daily = pd.Series(dtype=float)
    if ais_path.exists():
        ais_df = pd.read_csv(ais_path)
        if "time_bin" not in ais_df.columns or args.series not in ais_df.columns:
            raise KeyError(f"ais input must contain: time_bin, {args.series}")
        ais_daily = _resample_day(ais_df, "time_bin", args.series, fill_mode="none")
        has_ais_series = not ais_daily.empty
    elif not args.allow_missing_ais:
        raise FileNotFoundError(f"Missing ais input: {ais_path}")

    scale_factor = max(float(args.default_scale_factor), 0.0)
    scale_source = "default"
    scale_aligned_days = 0
    if has_ais_series:
        scale_factor, scale_aligned_days = _calc_scale(ais_daily=ais_daily, yolo_daily=yolo_daily)
        if scale_aligned_days > 0:
            scale_source = "ais_median_ratio"

    method_used = args.method
    if (
        not has_ais_series
        and method_used in {"seasonal", "seasonal_ma"}
        and yolo_observed_points < max(1, int(args.min_observed_points_for_seasonal))
    ):
        method_used = "ma"

    yolo_fc, freq = _forecast(
        series=yolo_daily,
        horizon=args.horizon,
        method=method_used,
        ma_window=args.ma_window,
        seasonal_period=args.seasonal_period,
    )
    yolo_ship_eq_fc = yolo_fc * scale_factor

    out = pd.DataFrame(
        {
            "time_bin": yolo_fc.index,
            "yolo_detections_forecast": yolo_fc.values,
            "yolo_ship_eq_forecast": yolo_ship_eq_fc.values,
        }
    )

    has_ais_forecast = False
    if ais_forecast_path.exists():
        ais_fc_df = pd.read_csv(ais_forecast_path)
        if "time_bin" in ais_fc_df.columns and "forecast_value" in ais_fc_df.columns:
            has_ais_forecast = True
            ais_fc_df["time_bin"] = pd.to_datetime(ais_fc_df["time_bin"])
            target_idx = ais_fc_df["time_bin"].dropna().drop_duplicates().sort_values()
            if not target_idx.empty:
                out = (
                    out.set_index("time_bin")
                    .sort_index()
                    .reindex(target_idx)
                )
                out[["yolo_detections_forecast", "yolo_ship_eq_forecast"]] = (
                    out[["yolo_detections_forecast", "yolo_ship_eq_forecast"]]
                    .interpolate(method="time")
                    .ffill()
                    .bfill()
                )
                out = out.reset_index().rename(columns={"index": "time_bin"})

            ais_fc_df = ais_fc_df.rename(columns={"forecast_value": "ais_forecast"})
            out = out.merge(ais_fc_df[["time_bin", "ais_forecast"]], on="time_bin", how="left")
        else:
            out["ais_forecast"] = pd.NA
    else:
        out["ais_forecast"] = pd.NA

    alpha = min(max(float(args.alpha), 0.0), 1.0)
    out["vision_forecast"] = out.apply(
        lambda r: float(r["yolo_ship_eq_forecast"])
        if pd.isna(r["ais_forecast"])
        else float(alpha * r["ais_forecast"] + (1.0 - alpha) * r["yolo_ship_eq_forecast"]),
        axis=1,
    )
    has_ais_blend = has_ais_forecast
    has_ais_calibration = scale_source == "ais_median_ratio"
    if has_ais_blend:
        semantic_unit = "vessels"
        confidence_level = "high" if yolo_observed_points >= 21 else "medium"
        confidence_reason = "Blended with AIS forecast."
    elif has_ais_calibration:
        semantic_unit = "vessel_equivalent_index"
        confidence_level = "medium"
        confidence_reason = "YOLO-only forecast calibrated by historical AIS ratio."
    else:
        semantic_unit = "detection_index"
        confidence_level = "low"
        confidence_reason = "YOLO-only forecast without AIS calibration."
    out["series"] = args.series
    out["method"] = method_used
    out["freq"] = freq
    out["scale_factor"] = scale_factor
    out["scale_source"] = scale_source
    out["scale_aligned_days"] = scale_aligned_days
    out["blend_alpha"] = alpha
    out["mode"] = "blend" if has_ais_forecast else "yolo_only"
    out["observed_points"] = yolo_observed_points
    out["semantic_unit"] = semantic_unit
    out["confidence_level"] = confidence_level
    out["confidence_reason"] = confidence_reason
    out["time_bin"] = pd.to_datetime(out["time_bin"]).dt.strftime("%Y-%m-%d")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"vision forecast -> {out_path}")


if __name__ == "__main__":
    main()
