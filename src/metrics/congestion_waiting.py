from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetricConfig:
    time_bin: str = "1H"
    sog_threshold: float = 0.5
    max_gap_min: int = 60
    min_duration_min: int = 20


def _normalize_time_bin(freq: str) -> str:
    # Pandas is deprecating upper-case aliases like "H".
    return str(freq).replace("H", "h")


def _load_clean_csvs(clean_dir: Path) -> pd.DataFrame:
    files = sorted(clean_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    parts: List[pd.DataFrame] = []
    for fp in files:
        try:
            parts.append(pd.read_csv(fp))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if "postime" not in df.columns:
        raise KeyError("AIS cleaned file must contain postime column")
    if "mmsi" not in df.columns:
        raise KeyError("AIS cleaned file must contain mmsi column")
    if "sog" not in df.columns:
        df["sog"] = np.nan
    return df


def build_congestion_curve(df: pd.DataFrame, cfg: MetricConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["time_bin", "presence_mmsi", "idle_mmsi"])
    data = df.copy()
    data["postime"] = pd.to_datetime(data["postime"], errors="coerce")
    data = data.dropna(subset=["postime"])
    data["mmsi"] = data["mmsi"].astype(str)
    data["sog_num"] = pd.to_numeric(data["sog"], errors="coerce")
    data["is_idle"] = data["sog_num"].le(cfg.sog_threshold)
    data["time_bin"] = data["postime"].dt.floor(_normalize_time_bin(cfg.time_bin))

    presence = (
        data.groupby("time_bin")["mmsi"]
        .nunique()
        .rename("presence_mmsi")
    )
    idle = (
        data[data["is_idle"]]
        .groupby("time_bin")["mmsi"]
        .nunique()
        .rename("idle_mmsi")
    )
    out = pd.concat([presence, idle], axis=1).fillna(0).reset_index()
    out["presence_mmsi"] = out["presence_mmsi"].astype(int)
    out["idle_mmsi"] = out["idle_mmsi"].astype(int)
    out = out.sort_values("time_bin")
    out["time_bin"] = pd.to_datetime(out["time_bin"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def _segment_idle_events(one_mmsi: pd.DataFrame, cfg: MetricConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    g = one_mmsi.sort_values("postime").copy()
    g["postime"] = pd.to_datetime(g["postime"], errors="coerce")
    g["sog_num"] = pd.to_numeric(g["sog"], errors="coerce")
    g = g.dropna(subset=["postime"])
    if g.empty:
        return []

    is_idle = g["sog_num"].le(cfg.sog_threshold).fillna(False).to_numpy()
    ts = g["postime"].to_numpy()
    events: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []

    start_idx = None
    last_idx = None
    for i in range(len(g)):
        if not is_idle[i]:
            if start_idx is not None and last_idx is not None:
                start_t = pd.Timestamp(ts[start_idx])
                end_t = pd.Timestamp(ts[last_idx])
                dur_min = int((end_t - start_t).total_seconds() // 60)
                if dur_min >= cfg.min_duration_min:
                    events.append((start_t, end_t, dur_min))
            start_idx = None
            last_idx = None
            continue

        if start_idx is None:
            start_idx = i
            last_idx = i
            continue

        prev_t = pd.Timestamp(ts[last_idx])
        cur_t = pd.Timestamp(ts[i])
        gap_min = (cur_t - prev_t).total_seconds() / 60.0
        if gap_min <= cfg.max_gap_min:
            last_idx = i
        else:
            start_t = pd.Timestamp(ts[start_idx])
            end_t = pd.Timestamp(ts[last_idx])
            dur_min = int((end_t - start_t).total_seconds() // 60)
            if dur_min >= cfg.min_duration_min:
                events.append((start_t, end_t, dur_min))
            start_idx = i
            last_idx = i

    if start_idx is not None and last_idx is not None:
        start_t = pd.Timestamp(ts[start_idx])
        end_t = pd.Timestamp(ts[last_idx])
        dur_min = int((end_t - start_t).total_seconds() // 60)
        if dur_min >= cfg.min_duration_min:
            events.append((start_t, end_t, dur_min))
    return events


def build_waiting_events(df: pd.DataFrame, cfg: MetricConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["mmsi", "start_time", "end_time", "duration_min", "points"])
    data = df.copy()
    data["mmsi"] = data["mmsi"].astype(str)
    rows = []
    for mmsi, g in data.groupby("mmsi"):
        evs = _segment_idle_events(g, cfg)
        for st, et, dur in evs:
            rows.append(
                {
                    "mmsi": mmsi,
                    "start_time": st.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": et.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_min": dur,
                    "points": int(len(g)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["mmsi", "start_time", "end_time", "duration_min", "points"])
    out = pd.DataFrame(rows).sort_values(["end_time", "mmsi"])
    return out


def _quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def build_waiting_by_day(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "count", "mean", "median", "p90", "p95", "max"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    e = events.copy()
    e["end_time"] = pd.to_datetime(e["end_time"], errors="coerce")
    e = e.dropna(subset=["end_time"])
    e["date"] = e["end_time"].dt.strftime("%Y-%m-%d")
    g = e.groupby("date")["duration_min"]
    out = pd.DataFrame(
        {
            "count": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "p90": g.quantile(0.90),
            "p95": g.quantile(0.95),
            "max": g.max(),
        }
    ).reset_index()
    return out[cols].sort_values("date")


def build_waiting_summary(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["count", "mean", "median", "p90", "p95", "max"]
    if events.empty:
        return pd.DataFrame([{k: 0 for k in cols}])
    s = pd.to_numeric(events["duration_min"], errors="coerce").dropna()
    row = {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p90": _quantile(s, 0.90),
        "p95": _quantile(s, 0.95),
        "max": float(s.max()) if not s.empty else 0.0,
    }
    return pd.DataFrame([row])


def run_metrics(
    clean_dir: Path,
    out_dir: Path,
    cfg: MetricConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_clean_csvs(clean_dir)
    curve = build_congestion_curve(df, cfg)
    events = build_waiting_events(df, cfg)
    by_day = build_waiting_by_day(events)
    summary = build_waiting_summary(events)

    curve.to_csv(out_dir / "congestion_curve.csv", index=False, encoding="utf-8-sig")
    events.to_csv(out_dir / "waiting_time_events.csv", index=False, encoding="utf-8-sig")
    by_day.to_csv(out_dir / "waiting_time_by_day.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "waiting_time_summary.csv", index=False, encoding="utf-8-sig")
