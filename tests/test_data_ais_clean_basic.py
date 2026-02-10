from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_ais_clean_basic_supports_recursive_input(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "data" / "interim" / "ais_clean" / "raw_tracks_csv" / "100"
    src.mkdir(parents=True, exist_ok=True)
    in_csv = src / "slice_aoi.csv"
    in_csv.write_text(
        "\n".join(
            [
                "mmsi,postime,lat,lon,sog",
                "100,2024-04-01 00:00:00,30.60,122.00,0.2",
                "100,bad_time,30.60,122.00,0.2",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["YANGSHAN_ROOT"] = str(root)
    cmd = [sys.executable, "src/data/ais_clean_basic.py"]
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    out_csv = root / "data" / "processed" / "ais_cleaned" / "raw_tracks_csv" / "100" / "slice_clean.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert str(df.iloc[0]["mmsi"]) == "100"
