from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], allow_fail: bool = False) -> bool:
    full = [sys.executable] + cmd
    print(f"[run] {' '.join(full)}")
    p = subprocess.run(full, cwd=str(ROOT))
    ok = p.returncode == 0
    if not ok and not allow_fail:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return ok


def _has_images(folder: Path) -> bool:
    if not folder.exists():
        return False
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return any(p.is_file() and p.suffix.lower() in exts for p in folder.rglob("*"))


def _has_ais_metrics() -> bool:
    return (ROOT / "outputs" / "metrics" / "congestion_curve.csv").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto pipeline for S1 quicklook + YOLO + vision forecast")
    parser.add_argument("--s1-dir", default="data/raw/s1")
    parser.add_argument("--s1-quicklook-dir", default="data/raw/s1/quicklook")
    parser.add_argument("--s1-grd-zip-dir", default="data/raw/s1/grd_zip")
    parser.add_argument("--s1-grd-png-dir", default="data/interim/s1_grd_png")
    parser.add_argument("--quicklook-dir", default="data/interim/s1_quicklook")
    parser.add_argument("--yolo-out", default="outputs/yolo")
    parser.add_argument("--model", default="")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--classes", default="")
    args = parser.parse_args()

    s1_dir = ROOT / args.s1_dir
    s1_quicklook_dir = ROOT / args.s1_quicklook_dir
    s1_grd_zip_dir = ROOT / args.s1_grd_zip_dir
    s1_grd_png_dir = ROOT / args.s1_grd_png_dir
    quicklook_dir = ROOT / args.quicklook_dir

    if s1_dir.exists() and any((ROOT / args.s1_dir).glob("*.dat")):
        _run(
            [
                "scripts/prepare_s1_quicklook.py",
                "--input-dir",
                args.s1_dir,
                "--output-dir",
                args.quicklook_dir,
            ],
            allow_fail=True,
        )
    else:
        print(f"[warn] no .dat in {s1_dir}, skip dat->quicklook conversion")

    if s1_quicklook_dir.exists():
        _run(
            [
                "scripts/sync_s1_quicklook.py",
                "--src",
                args.s1_quicklook_dir,
                "--dst",
                args.quicklook_dir,
                "--mode",
                "copy",
            ],
            allow_fail=True,
        )

    if s1_grd_zip_dir.exists() and any(s1_grd_zip_dir.glob("*.zip")):
        _run(
            [
                "scripts/prepare_s1_grd.py",
                "--input-dir",
                args.s1_grd_zip_dir,
                "--output-dir",
                args.s1_grd_png_dir,
            ],
            allow_fail=True,
        )
    else:
        print(f"[warn] no .zip in {s1_grd_zip_dir}, skip GRD->PNG conversion")

    if not _has_images(quicklook_dir):
        print("[warn] no quicklook images available")

    yolo_input_dir = quicklook_dir
    if _has_images(s1_grd_png_dir):
        yolo_input_dir = s1_grd_png_dir
    elif not _has_images(quicklook_dir):
        print("[warn] no YOLO input images available, stop")
        return
    print(f"[info] YOLO input dir: {yolo_input_dir}")

    # Prefer SAR-specific tuned model if available.
    model_to_use = args.model.strip() or "yolov8n.pt"
    sar_model = ROOT / "assets" / "models" / "sar_ship_yolov8n.pt"
    if not args.model.strip() and sar_model.exists():
        model_to_use = str(sar_model)

    yolo_cmd = [
        "scripts/run_yolo.py",
        "--input",
        str(yolo_input_dir.relative_to(ROOT)),
        "--output",
        args.yolo_out,
        "--model",
        model_to_use,
        "--conf",
        "0.10",
        "--tiled",
        "--tile-size",
        "1024",
        "--tile-overlap",
        "0.2",
    ]
    if args.classes.strip():
        yolo_cmd += ["--classes", args.classes]
    _run(yolo_cmd)
    _run(["scripts/build_yolo_observed.py", "--yolo-dir", args.yolo_out])

    if not _has_ais_metrics():
        print("[warn] AIS metrics not ready, skip vision forecast for now")
        print("[done] yolo detections and yolo_observed are ready")
        return

    # Prefer interpolation; fallback to ffill to avoid empty series.
    ok = _run(
        [
            "scripts/run_vision_forecast.py",
            "--horizon",
            str(args.horizon),
            "--yolo-fill-mode",
            "interpolate",
        ],
        allow_fail=True,
    )
    if not ok:
        _run(
            [
                "scripts/run_vision_forecast.py",
                "--horizon",
                str(args.horizon),
                "--yolo-fill-mode",
                "ffill",
            ]
        )
    print("[done] s1+yolo auto pipeline completed")


if __name__ == "__main__":
    main()
