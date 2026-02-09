from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_vision_forecast_yolo_only(tmp_path: Path) -> None:
    yolo_csv = tmp_path / "yolo_observed.csv"
    out_csv = tmp_path / "vision_forecast.csv"
    missing_ais = tmp_path / "missing_congestion_curve.csv"
    missing_ais_fc = tmp_path / "missing_congestion_forecast.csv"

    pd.DataFrame(
        [
            {"time_bin": "2024-04-01 09:55:00", "yolo_detections": 10, "yolo_files": 1, "yolo_mean_conf": 0.7},
            {"time_bin": "2024-04-02 09:55:00", "yolo_detections": 12, "yolo_files": 1, "yolo_mean_conf": 0.8},
            {"time_bin": "2024-04-03 09:55:00", "yolo_detections": 14, "yolo_files": 1, "yolo_mean_conf": 0.75},
        ]
    ).to_csv(yolo_csv, index=False, encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_vision_forecast.py",
        "--yolo-input",
        str(yolo_csv),
        "--ais-input",
        str(missing_ais),
        "--ais-forecast-input",
        str(missing_ais_fc),
        "--output",
        str(out_csv),
        "--horizon",
        "5",
        "--method",
        "naive",
        "--allow-missing-ais",
    ]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.exists()

    out = pd.read_csv(out_csv)
    assert len(out) == 5
    assert set(
        [
            "vision_forecast",
            "yolo_ship_eq_forecast",
            "mode",
            "scale_factor",
            "semantic_unit",
            "confidence_level",
            "confidence_reason",
            "scale_source",
            "scale_aligned_days",
        ]
    ).issubset(out.columns)
    assert out["mode"].nunique() == 1
    assert out["mode"].iloc[0] == "yolo_only"
    assert float(out["scale_factor"].iloc[0]) == 1.0
    assert out["semantic_unit"].iloc[0] == "detection_index"
    assert out["confidence_level"].iloc[0] == "low"
    assert out["scale_source"].iloc[0] == "default"
    assert int(out["scale_aligned_days"].iloc[0]) == 0

    vf = pd.to_numeric(out["vision_forecast"], errors="coerce")
    yf = pd.to_numeric(out["yolo_ship_eq_forecast"], errors="coerce")
    assert vf.notna().all()
    assert yf.notna().all()
    assert (vf.round(6) == yf.round(6)).all()
