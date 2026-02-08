from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ZIP_URL = "https://zenodo.org/records/13851569/files/ultralytics-20240333.zip?download=1"
ZIP_PATH = ROOT / "assets" / "models" / "ultralytics-20240333.zip"
EXTRACT_ROOT = ROOT / "data" / "interim" / "sar_ship_dataset"
FINAL_MODEL = ROOT / "assets" / "models" / "sar_ship_yolov8n.pt"


def _download_zip() -> None:
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        print(f"[zip] exists: {ZIP_PATH}")
        return
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    import requests

    print(f"[zip] downloading: {ZIP_URL}")
    with requests.get(ZIP_URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with ZIP_PATH.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    print(f"[zip] downloaded: {ZIP_PATH}")


def _extract_dataset(force: bool = False) -> Path:
    target = EXTRACT_ROOT
    if force and target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if target.exists() and (target / "dataset").exists():
        print(f"[extract] exists: {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = [m for m in zf.namelist() if "ultralytics-main/dataset/" in m]
        if not members:
            raise FileNotFoundError("dataset folder not found in zip")
        for m in members:
            # Keep path after "ultralytics-main/" to simplify local layout.
            idx = m.find("ultralytics-main/")
            rel = m[idx + len("ultralytics-main/") :]
            out = target / rel
            if m.endswith("/"):
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, out.open("wb") as dst:
                dst.write(src.read())
    print(f"[extract] done: {target}")
    return target


def _prepare_data_yaml(dataset_root: Path) -> Path:
    src_yaml = dataset_root / "dataset" / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(f"missing {src_yaml}")

    payload = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    payload["path"] = str(dataset_root.resolve())
    payload["train"] = "dataset/images/train"
    payload["val"] = "dataset/images/val"
    payload["test"] = "dataset/images/test"

    out_yaml = dataset_root / "data_local.yaml"
    out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[yaml] ready: {out_yaml}")
    return out_yaml


def _train(
    data_yaml: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    run_name: str,
) -> Path:
    from ultralytics import YOLO

    project = ROOT / "outputs" / "train"
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_model)
    print(
        f"[train] model={base_model} epochs={epochs} imgsz={imgsz} "
        f"batch={batch} device={device} workers={workers}"
    )
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project),
        name=run_name,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found under {save_dir}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAR-ship-specific YOLO model from Zenodo dataset")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-name", default="sar_ship_yolov8n")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    _download_zip()
    dataset_root = _extract_dataset(force=args.force_extract)
    data_yaml = _prepare_data_yaml(dataset_root)
    best = _train(
        data_yaml=data_yaml,
        base_model=args.base_model,
        epochs=max(1, args.epochs),
        imgsz=max(320, args.imgsz),
        batch=max(1, args.batch),
        device=args.device,
        workers=max(0, args.workers),
        run_name=args.run_name,
    )

    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, FINAL_MODEL)
    print(f"[done] best={best}")
    print(f"[done] copied={FINAL_MODEL}")


if __name__ == "__main__":
    main()

