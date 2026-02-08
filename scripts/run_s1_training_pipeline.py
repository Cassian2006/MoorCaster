from __future__ import annotations

import argparse
import json
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
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if not folder.exists():
        return False
    return any(p.is_file() and p.suffix.lower() in exts for p in folder.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end S1 -> YOLO pseudo dataset -> fine-tune pipeline")
    parser.add_argument("--s1-grd-zip-dir", default="data/raw/s1/grd_zip")
    parser.add_argument("--s1-grd-png-dir", default="data/interim/s1_grd_png")
    parser.add_argument("--yolo-output", default="outputs/yolo")
    parser.add_argument("--dataset-dir", default="data/interim/s1_yolo_pseudo")
    parser.add_argument("--model", default="")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--pseudo-conf", type=float, default=0.35)
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--min-labeled-images", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-name", default="s1_pseudo_finetune")
    parser.add_argument("--export-model", default="assets/models/sar_ship_yolov8n_s1.pt")
    args = parser.parse_args()

    s1_grd_zip_dir = ROOT / args.s1_grd_zip_dir
    s1_grd_png_dir = ROOT / args.s1_grd_png_dir

    if s1_grd_zip_dir.exists() and any(s1_grd_zip_dir.glob("*.zip")):
        _run(
            [
                "scripts/prepare_s1_grd.py",
                "--input-dir",
                args.s1_grd_zip_dir,
                "--output-dir",
                args.s1_grd_png_dir,
            ]
        )
    if not _has_images(s1_grd_png_dir):
        raise RuntimeError(f"no S1 images ready in {s1_grd_png_dir}")

    model_to_use = args.model.strip()
    if not model_to_use:
        preferred = ROOT / "assets" / "models" / "sar_ship_yolov8n.pt"
        model_to_use = str(preferred) if preferred.exists() else "yolov8n.pt"

    if not args.skip_yolo:
        _run(
            [
                "scripts/run_yolo.py",
                "--input",
                args.s1_grd_png_dir,
                "--output",
                args.yolo_output,
                "--model",
                model_to_use,
                "--conf",
                str(args.yolo_conf),
                "--tiled",
                "--tile-size",
                "1024",
                "--tile-overlap",
                "0.2",
            ]
        )

    _run(
        [
            "scripts/build_s1_pseudo_yolo_dataset.py",
            "--image-dir",
            args.s1_grd_png_dir,
            "--detection-dir",
            args.yolo_output,
            "--output-dir",
            args.dataset_dir,
            "--min-conf",
            str(args.pseudo_conf),
            "--clean",
        ]
        + (["--keep-empty"] if args.keep_empty else [])
    )

    summary_path = ROOT / args.dataset_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labeled_images = int(summary.get("labeled_images", 0))
    print(
        "[summary] "
        f"kept_images={summary.get('kept_images', 0)} "
        f"labeled_images={labeled_images} "
        f"boxes={summary.get('boxes', 0)}"
    )

    if args.skip_train:
        print("[done] training skipped by flag")
        return
    if labeled_images < max(1, int(args.min_labeled_images)):
        print(
            "[warn] labeled_images below threshold, skip train: "
            f"{labeled_images} < {int(args.min_labeled_images)}"
        )
        return

    _run(
        [
            "scripts/train_s1_yolo_finetune.py",
            "--data-yaml",
            str((ROOT / args.dataset_dir / "data.yaml").resolve()),
            "--base-model",
            model_to_use,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            str(args.device),
            "--workers",
            str(args.workers),
            "--run-name",
            args.run_name,
            "--export-model",
            args.export_model,
        ]
    )
    print("[done] s1 training pipeline completed")


if __name__ == "__main__":
    main()

