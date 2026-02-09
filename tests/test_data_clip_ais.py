from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.clip_ais import clip_ais


def test_clip_ais_recursive_and_skip_non_geo(tmp_path: Path) -> None:
    root = tmp_path
    raw = root / "data" / "raw" / "ais"
    raw.mkdir(parents=True, exist_ok=True)

    # Non-geo helper CSV should not break clipping.
    (raw / "mmsi.csv").write_text("mmsi,count\n100,2\n", encoding="utf-8")

    track_dir = raw / "raw_tracks_csv" / "100"
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "slice.csv").write_text(
        "\n".join(
            [
                "mmsi,postime,lat,lon,sog",
                "100,2024-04-01 00:00:00,30.60,122.00,0.2",
                "100,2024-04-01 00:10:00,31.00,122.00,0.1",
            ]
        ),
        encoding="utf-8",
    )

    clip_ais(root)

    out = root / "data" / "interim" / "ais_clean" / "raw_tracks_csv" / "100" / "slice.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 1
    assert float(df.iloc[0]["lat"]) == 30.60
    assert not (root / "data" / "interim" / "ais_clean" / "mmsi.csv").exists()
