import os
from pathlib import Path

import pandas as pd


def clip_ais(root: Path, aoi_lat=(30.50, 30.75), aoi_lon=(121.90, 122.25)):
    raw_dir = root / "data" / "raw" / "ais"
    out_dir = root / "data" / "interim" / "ais_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(p for p in raw_dir.rglob("*.csv") if p.is_file())
    for csv_path in csv_files:
        rel = csv_path.relative_to(raw_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        wrote_header = False
        total_in = 0
        total_out = 0
        skipped_no_geo = False

        for chunk in pd.read_csv(csv_path, chunksize=200_000):
            total_in += len(chunk)
            if "lat" not in chunk.columns or "lon" not in chunk.columns:
                skipped_no_geo = True
                break
            lat = pd.to_numeric(chunk["lat"], errors="coerce")
            lon = pd.to_numeric(chunk["lon"], errors="coerce")
            mask = (
                lat.ge(aoi_lat[0])
                & lat.le(aoi_lat[1])
                & lon.ge(aoi_lon[0])
                & lon.le(aoi_lon[1])
            )
            sub = chunk.loc[mask]
            total_out += len(sub)
            if not sub.empty:
                sub.to_csv(out_path, mode="a", index=False, header=not wrote_header)
                wrote_header = True

        if skipped_no_geo:
            print(f"{csv_path.name}: skipped (missing lat/lon)")
            continue
        print(f"{csv_path.name}: {total_out}/{total_in} rows kept -> {out_path}")


def main():
    root = Path(os.environ.get("YANGSHAN_ROOT", Path(__file__).resolve().parents[2]))
    clip_ais(root)


if __name__ == "__main__":
    main()
