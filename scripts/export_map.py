from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _pick_source() -> Path | None:
    for folder in (
        ROOT / "data" / "processed" / "ais_cleaned",
        ROOT / "data" / "interim" / "ais_clean",
        ROOT / "data" / "raw" / "ais",
    ):
        files = sorted(folder.glob("*.csv"))
        if files:
            return files[0]
    return None


def _sample_points(src: Path, limit: int) -> List[dict]:
    rows: List[dict] = []
    for chunk in pd.read_csv(src, chunksize=50000):
        if "lon" not in chunk.columns or "lat" not in chunk.columns:
            continue
        sub = chunk[["lon", "lat"] + [c for c in ("mmsi", "postime", "sog") if c in chunk.columns]].copy()
        sub["lon"] = pd.to_numeric(sub["lon"], errors="coerce")
        sub["lat"] = pd.to_numeric(sub["lat"], errors="coerce")
        sub = sub.dropna(subset=["lon", "lat"])
        rows.extend(sub.to_dict(orient="records"))
        if len(rows) >= limit:
            break
    if len(rows) > limit:
        step = max(len(rows) // limit, 1)
        rows = rows[::step][:limit]
    return rows


def _html(points: List[dict]) -> str:
    center_lat = 30.625
    center_lon = 122.075
    payload = json.dumps(points, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AIS Points Quality Check</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    body {{ margin: 0; font-family: "Segoe UI", sans-serif; background: #0c111f; color: #e8ecff; }}
    #head {{ padding: 10px 14px; border-bottom: 1px solid #1f2a44; }}
    #map {{ width: 100vw; height: calc(100vh - 48px); }}
    .meta {{ color: #9eb1de; font-size: 13px; }}
  </style>
</head>
<body>
  <div id="head">
    <div><strong>AIS Points Map</strong> <span class="meta">(quality check only)</span></div>
    <div class="meta">ROI: lon 121.90~122.25, lat 30.50~30.75</div>
  </div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {payload};
    const map = L.map('map').setView([{center_lat}, {center_lon}], 11);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const roi = L.rectangle([[30.50, 121.90], [30.75, 122.25]], {{
      color: '#4ecdc4',
      weight: 2,
      fillOpacity: 0.06
    }}).addTo(map);
    roi.bindTooltip('Yangshan ROI');

    points.forEach(p => {{
      const lat = Number(p.lat);
      const lon = Number(p.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const marker = L.circleMarker([lat, lon], {{
        radius: 2,
        color: '#ff7a59',
        fillOpacity: 0.7,
        weight: 0
      }}).addTo(map);
      marker.bindPopup(`MMSI: ${{p.mmsi || ''}}<br>Time: ${{p.postime || ''}}<br>SOG: ${{p.sog ?? ''}}`);
    }});
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a lightweight AIS points HTML map for quality check")
    parser.add_argument("--output", default="outputs/export/ais_points_map.html")
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()

    src = _pick_source()
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src:
        out_path.write_text("<html><body><h3>No AIS csv found.</h3></body></html>", encoding="utf-8")
        print(f"[done] no source csv found -> {out_path}")
        return

    points = _sample_points(src, max(100, min(args.limit, 10000)))
    out_path.write_text(_html(points), encoding="utf-8")
    print(f"[done] map={out_path} source={src} points={len(points)}")


if __name__ == "__main__":
    main()

