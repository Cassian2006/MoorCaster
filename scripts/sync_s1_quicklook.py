from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync S1 quicklook images into YOLO input directory")
    parser.add_argument("--src", default="data/raw/s1/quicklook")
    parser.add_argument("--dst", default="data/interim/s1_quicklook")
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[sync] source not found: {src}")
        return

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = sorted([p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not files:
        print(f"[sync] no images in {src}")
        return

    copied = 0
    skipped = 0
    for fp in files:
        out = dst / fp.name
        if out.exists():
            if not args.overwrite:
                skipped += 1
                continue
            out.unlink()
        if args.mode == "hardlink":
            try:
                out.hardlink_to(fp)
            except Exception:
                shutil.copy2(fp, out)
        else:
            shutil.copy2(fp, out)
        copied += 1

    print(f"[sync] src={src} dst={dst} found={len(files)} copied={copied} skipped={skipped}")


if __name__ == "__main__":
    main()

