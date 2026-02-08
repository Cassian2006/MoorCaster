from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _resolve_model(model_arg: str) -> str:
    if not model_arg.strip():
        preferred = ROOT / "assets" / "models" / "sar_ship_yolov8n.pt"
        if preferred.exists():
            return str(preferred)
        return "yolov8n.pt"
    p = Path(model_arg)
    if p.exists():
        return str(p)
    p2 = ROOT / model_arg
    if p2.exists():
        return str(p2)
    # Allow ultralytics to auto-download canonical model names.
    return model_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model on local S1 pseudo-labeled dataset")
    parser.add_argument("--data-yaml", default="data/interim/s1_yolo_pseudo/data.yaml")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="outputs/train")
    parser.add_argument("--run-name", default="s1_pseudo_finetune")
    parser.add_argument("--export-model", default="assets/models/sar_ship_yolov8n_s1.pt")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"missing data yaml: {data_yaml}")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency: ultralytics. Install requirements-vision.txt first.") from exc

    base_model = _resolve_model(args.base_model)
    print(
        f"[train] base_model={base_model} data={data_yaml} "
        f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}"
    )

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=max(1, int(args.epochs)),
        imgsz=max(320, int(args.imgsz)),
        batch=max(1, int(args.batch)),
        device=args.device,
        workers=max(0, int(args.workers)),
        patience=max(1, int(args.patience)),
        project=str((ROOT / args.project).resolve()),
        name=args.run_name,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found under {save_dir}")

    export_path = ROOT / args.export_model
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_path)
    print(f"[done] best={best}")
    print(f"[done] exported={export_path}")


if __name__ == "__main__":
    main()

