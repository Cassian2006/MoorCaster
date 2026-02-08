from __future__ import annotations

import pandas as pd

from src.metrics.congestion_waiting import (
    MetricConfig,
    build_congestion_curve,
    build_waiting_by_day,
    build_waiting_events,
    build_waiting_summary,
)


def test_build_congestion_curve_counts_presence_and_idle() -> None:
    df = pd.DataFrame(
        [
            {"mmsi": "100", "postime": "2024-04-01 00:10:00", "sog": 0.2},
            {"mmsi": "100", "postime": "2024-04-01 00:20:00", "sog": 1.0},
            {"mmsi": "200", "postime": "2024-04-01 00:15:00", "sog": 0.1},
            {"mmsi": "300", "postime": "2024-04-01 01:05:00", "sog": 0.4},
        ]
    )
    cfg = MetricConfig(time_bin="1H", sog_threshold=0.5, max_gap_min=60, min_duration_min=20)
    out = build_congestion_curve(df, cfg)

    assert list(out["time_bin"]) == ["2024-04-01 00:00:00", "2024-04-01 01:00:00"]
    assert list(out["presence_mmsi"]) == [2, 1]
    assert list(out["idle_mmsi"]) == [2, 1]


def test_build_waiting_events_respects_gap_and_min_duration() -> None:
    df = pd.DataFrame(
        [
            {"mmsi": "100", "postime": "2024-04-01 00:00:00", "sog": 0.1},
            {"mmsi": "100", "postime": "2024-04-01 00:10:00", "sog": 0.2},
            {"mmsi": "100", "postime": "2024-04-01 00:20:00", "sog": 0.2},
            {"mmsi": "100", "postime": "2024-04-01 00:30:00", "sog": 1.2},
            {"mmsi": "100", "postime": "2024-04-01 01:00:00", "sog": 0.1},
            {"mmsi": "100", "postime": "2024-04-01 02:30:00", "sog": 0.1},
            {"mmsi": "200", "postime": "2024-04-01 00:00:00", "sog": 0.3},
            {"mmsi": "200", "postime": "2024-04-01 00:50:00", "sog": 0.4},
        ]
    )
    cfg = MetricConfig(time_bin="1H", sog_threshold=0.5, max_gap_min=60, min_duration_min=20)
    events = build_waiting_events(df, cfg)

    assert len(events) == 2
    durations = sorted(events["duration_min"].tolist())
    assert durations == [20, 50]


def test_waiting_aggregates_with_empty_and_non_empty() -> None:
    empty_events = pd.DataFrame(columns=["mmsi", "start_time", "end_time", "duration_min", "points"])
    by_day_empty = build_waiting_by_day(empty_events)
    summary_empty = build_waiting_summary(empty_events)

    assert by_day_empty.empty
    assert summary_empty.iloc[0]["count"] == 0
    assert summary_empty.iloc[0]["p90"] == 0

    events = pd.DataFrame(
        [
            {"mmsi": "1", "start_time": "2024-04-01 00:00:00", "end_time": "2024-04-01 01:00:00", "duration_min": 60, "points": 10},
            {"mmsi": "2", "start_time": "2024-04-01 02:00:00", "end_time": "2024-04-01 03:30:00", "duration_min": 90, "points": 11},
        ]
    )
    by_day = build_waiting_by_day(events)
    summary = build_waiting_summary(events)

    assert by_day.iloc[0]["count"] == 2
    assert round(float(by_day.iloc[0]["mean"]), 2) == 75.00
    assert int(summary.iloc[0]["max"]) == 90
