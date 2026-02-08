import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling


def _scale_to_uint8(arr: np.ndarray) -> np.ndarray:
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max()) if float(valid.max()) > lo else lo + 1.0
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _target_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    longest = max(height, width)
    if longest <= max_size:
        return height, width
    scale = max_size / float(longest)
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Build downsampled PNG quicklooks from Sentinel-1 GeoTIFF .dat files")
    parser.add_argument("--input-dir", default="data/raw/s1")
    parser.add_argument("--output-dir", default="data/interim/s1_quicklook")
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.dat"))
    if not files:
        print(f"[quicklook] no .dat found in {input_dir}")
        return

    for i, fp in enumerate(files, 1):
        out = output_dir / f"{fp.stem}.png"
        if out.exists() and not args.overwrite:
            print(f"[skip {i}/{len(files)}] {out.name}")
            continue

        with rasterio.open(fp) as ds:
            out_h, out_w = _target_shape(ds.height, ds.width, args.max_size)
            band = ds.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        gray = _scale_to_uint8(band)
        rgb = np.stack([gray, gray, gray], axis=2)
        Image.fromarray(rgb, mode="RGB").save(out)
        print(f"[ok {i}/{len(files)}] {fp.name} -> {out.name} ({out_w}x{out_h})")

    print(f"[done] quicklooks -> {output_dir}")


if __name__ == "__main__":
    main()

