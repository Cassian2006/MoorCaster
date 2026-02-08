from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    full = [sys.executable] + cmd
    print(f"[run] {' '.join(full)}")
    proc = subprocess.run(full, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def _exists_any(folder: Path, suffix: str = ".csv") -> bool:
    return folder.exists() and any(folder.glob(f"*{suffix}"))


def _has_images(folder: Path) -> bool:
    if not folder.exists():
        return False
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return any(p.is_file() and p.suffix.lower() in exts for p in folder.rglob("*"))


def _is_model_usable(model_arg: str) -> bool:
    p = ROOT / model_arg
    if p.exists():
        return True
    # Ultralytics can auto-download common model names like yolov8n.pt.
    if "/" not in model_arg and "\\" not in model_arg and model_arg.lower().endswith(".pt"):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end local pipeline runner")
    parser.add_argument("--horizon-days", type=int, default=24)
    parser.add_argument("--time-bin", default="1H")
    parser.add_argument("--sog-threshold", type=float, default=0.5)
    parser.add_argument("--max-gap-min", type=int, default=60)
    parser.add_argument("--min-duration-min", type=int, default=20)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--yolo-classes", default="")
    parser.add_argument("--yolo-input", default="data/interim/s1_quicklook")
    parser.add_argument("--yolo-grd-input", default="data/interim/s1_grd_png")
    parser.add_argument("--yolo-output", default="outputs/yolo")
    args = parser.parse_args()

    raw_ais = ROOT / "data" / "raw" / "ais"
    interim_ais = ROOT / "data" / "interim" / "ais_clean"
    processed_ais = ROOT / "data" / "processed" / "ais_cleaned"
    metrics_dir = ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_clip:
        if _exists_any(raw_ais):
            _run(["src/data/clip_ais.py"])
        else:
            print("[warn] skip clip_ais: no raw csv found in data/raw/ais")

    if not args.skip_clean:
        if _exists_any(interim_ais):
            _run(["src/data/ais_clean_basic.py"])
        else:
            print("[warn] skip clean_ais: no clipped csv found in data/interim/ais_clean")

    if _exists_any(processed_ais):
        _run(
            [
                "scripts/run_metrics.py",
                "--time-bin",
                args.time_bin,
                "--sog-threshold",
                str(args.sog_threshold),
                "--max-gap-min",
                str(args.max_gap_min),
                "--min-duration-min",
                str(args.min_duration_min),
            ]
        )
        _run(["scripts/run_forecast.py", "--horizon", str(args.horizon_days)])
        _run(["scripts/run_waiting_forecast.py", "--horizon", str(args.horizon_days)])
    else:
        print("[warn] metrics/forecast skipped: no cleaned AIS csv found")

    if not args.skip_yolo:
        _run(
            [
                "scripts/prepare_s1_grd.py",
                "--input-dir",
                "data/raw/s1/grd_zip",
                "--output-dir",
                args.yolo_grd_input,
            ]
        )
        # Keep YOLO input in sync with downloaded S1 quicklooks.
        _run(
            [
                "scripts/sync_s1_quicklook.py",
                "--src",
                "data/raw/s1/quicklook",
                "--dst",
                args.yolo_input,
                "--mode",
                "copy",
            ]
        )
        yolo_input = ROOT / args.yolo_input
        yolo_grd_input = ROOT / args.yolo_grd_input
        if _has_images(yolo_grd_input):
            yolo_input = yolo_grd_input
        model_to_use = args.yolo_model.strip()
        sar_model = ROOT / "assets" / "models" / "sar_ship_yolov8n.pt"
        if not model_to_use:
            model_to_use = str(sar_model) if sar_model.exists() else "yolov8n.pt"
        if _has_images(yolo_input) and _is_model_usable(model_to_use):
            _run(
                [
                    "scripts/run_yolo.py",
                    "--input",
                    args.yolo_input,
                    "--output",
                    args.yolo_output,
                    "--model",
                    model_to_use,
                ]
                + (["--classes", args.yolo_classes] if args.yolo_classes.strip() else [])
            )
            _run(["scripts/build_yolo_observed.py", "--yolo-dir", args.yolo_output])
            _run(
                [
                    "scripts/run_vision_forecast.py",
                    "--horizon",
                    str(args.horizon_days),
                    "--yolo-fill-mode",
                    "interpolate",
                ]
            )
        else:
            print("[warn] YOLO skipped: missing quicklook images or model")

    _run(["scripts/build_evidence_cards.py"])
    _run(["scripts/export_map.py"])
    print("[done] pipeline completed")


if __name__ == "__main__":
    main()
