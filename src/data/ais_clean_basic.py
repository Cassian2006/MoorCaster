import json
import os
from pathlib import Path

import pandas as pd


def clean_ais_file(
    src_path: Path,
    dst_path: Path,
    min_lat=30.50,
    max_lat=30.75,
    min_lon=121.90,
    max_lon=122.25,
    min_sog=0.0,
    max_sog=50.0,
):
    stats = {
        "file": str(src_path),
        "rows_in": 0,
        "rows_out": 0,
        "time_parse_ok_ratio": None,
        "invalid_latlon": 0,
        "invalid_sog": 0,
        "invalid_time": 0,
        "duplicate_rows": 0,
    }

    wrote_header = False
    time_ok = 0
    time_total = 0

    for chunk in pd.read_csv(src_path, chunksize=200_000):
        stats["rows_in"] += len(chunk)

        # Required columns
        for col in ["mmsi", "postime", "lat", "lon"]:
            if col not in chunk.columns:
                raise KeyError(f"{col} column missing in {src_path.name}")

        lat = pd.to_numeric(chunk["lat"], errors="coerce")
        lon = pd.to_numeric(chunk["lon"], errors="coerce")

        lat_ok = lat.between(min_lat, max_lat)
        lon_ok = lon.between(min_lon, max_lon)
        latlon_ok = lat_ok & lon_ok

        stats["invalid_latlon"] += int((~latlon_ok).sum())

        if "sog" in chunk.columns:
            sog = pd.to_numeric(chunk["sog"], errors="coerce")
            sog_ok = sog.between(min_sog, max_sog)
            stats["invalid_sog"] += int((~sog_ok).sum())
        else:
            sog_ok = pd.Series(True, index=chunk.index)

        t = pd.to_datetime(chunk["postime"], errors="coerce")
        time_total += len(t)
        time_ok += int(t.notna().sum())
        time_ok_mask = t.notna()
        stats["invalid_time"] += int((~time_ok_mask).sum())

        cleaned = chunk.loc[latlon_ok & sog_ok & time_ok_mask].copy()

        # De-duplicate on common identifiers within chunk
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["mmsi", "postime", "lat", "lon"])
        stats["duplicate_rows"] += int(before - len(cleaned))

        if not cleaned.empty:
            cleaned.to_csv(dst_path, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            stats["rows_out"] += len(cleaned)

    if time_total > 0:
        stats["time_parse_ok_ratio"] = time_ok / time_total
    return stats


def main():
    root = Path(os.environ.get("YANGSHAN_ROOT", Path(__file__).resolve().parents[2]))
    src_dir = root / "data" / "interim" / "ais_clean"
    out_dir = root / "data" / "processed" / "ais_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"files": []}
    for csv_path in sorted(src_dir.glob("*.csv")):
        dst_path = out_dir / csv_path.name.replace("_aoi.csv", "_clean.csv")
        if dst_path.exists():
            dst_path.unlink()
        stats = clean_ais_file(csv_path, dst_path)
        report["files"].append(stats)
        print(f"{csv_path.name}: {stats['rows_out']}/{stats['rows_in']} kept -> {dst_path}")

    report_path = root / "outputs" / "metrics" / "ais_clean_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
