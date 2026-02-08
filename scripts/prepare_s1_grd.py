from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling


def _target_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    longest = max(height, width)
    if longest <= max_size:
        return height, width
    scale = max_size / float(longest)
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def _scale_sar_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    # Convert SAR amplitude-like values to log domain for better contrast.
    db = 10.0 * np.log10(arr + 1e-6)
    valid = db[np.isfinite(db)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _pick_measurement(zf: zipfile.ZipFile, prefer_pol: str = "vv") -> Optional[str]:
    names = zf.namelist()
    tif_names = [n for n in names if n.lower().endswith((".tif", ".tiff")) and "/measurement/" in n.lower()]
    if not tif_names:
        return None
    prefer_pol = prefer_pol.lower().strip()
    preferred = [n for n in tif_names if f"-{prefer_pol}-" in n.lower() or f"_{prefer_pol}_" in n.lower()]
    if preferred:
        return sorted(preferred)[0]
    return sorted(tif_names)[0]


def _zip_to_png(zip_path: Path, output_path: Path, max_size: int, prefer_pol: str) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        measurement = _pick_measurement(zf, prefer_pol=prefer_pol)
        if not measurement:
            print(f"[skip] no measurement tiff in {zip_path.name}")
            return False
        with tempfile.TemporaryDirectory() as td:
            extracted = Path(zf.extract(measurement, path=td))
            with rasterio.open(extracted) as ds:
                out_h, out_w = _target_shape(ds.height, ds.width, max_size=max_size)
                band = ds.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
    gray = _scale_sar_to_uint8(band)
    rgb = np.stack([gray, gray, gray], axis=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"[ok] {zip_path.name} -> {output_path.name} ({out_w}x{out_h})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO-ready PNGs from Sentinel-1 GRD product ZIPs")
    parser.add_argument("--input-dir", default="data/raw/s1/grd_zip")
    parser.add_argument("--output-dir", default="data/interim/s1_grd_png")
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--prefer-pol", default="vv", help="Preferred polarization, e.g. vv or vh")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        print(f"[prepare_s1_grd] no .zip files found in {input_dir}")
        return

    ok = 0
    for idx, zp in enumerate(zips, 1):
        out = output_dir / f"{zp.stem}_{args.prefer_pol.lower()}.png"
        if out.exists() and not args.overwrite:
            print(f"[skip {idx}/{len(zips)}] {out.name}")
            continue
        try:
            if _zip_to_png(zp, out, max_size=args.max_size, prefer_pol=args.prefer_pol):
                ok += 1
        except Exception as exc:
            print(f"[error {idx}/{len(zips)}] {zp.name}: {exc}")
    print(f"[done] prepared={ok} total_zip={len(zips)} out_dir={output_dir}")


if __name__ == "__main__":
    main()
