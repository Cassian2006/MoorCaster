from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


def _iter_images(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _xyxy_to_yolo_line(
    xyxy: List[float],
    img_w: int,
    img_h: int,
    cls_id: int,
) -> str:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = _clip(x1, 0.0, float(img_w - 1))
    x2 = _clip(x2, 0.0, float(img_w - 1))
    y1 = _clip(y1, 0.0, float(img_h - 1))
    y2 = _clip(y2, 0.0, float(img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return ""

    xc = (x1 + x2) / 2.0 / float(img_w)
    yc = (y1 + y2) / 2.0 / float(img_h)
    bw = (x2 - x1) / float(img_w)
    bh = (y2 - y1) / float(img_h)
    xc = _clip(xc, 0.0, 1.0)
    yc = _clip(yc, 0.0, 1.0)
    bw = _clip(bw, 0.0, 1.0)
    bh = _clip(bh, 0.0, 1.0)
    if bw <= 0 or bh <= 0:
        return ""
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _load_detections(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    rows: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        xyxy = row.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        rows.append(row)
    return rows


def _split(items: List[Path], train_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    arr = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(arr)
    cut = int(round(len(arr) * train_ratio))
    cut = max(1, min(len(arr) - 1, cut)) if len(arr) >= 2 else len(arr)
    return arr[:cut], arr[cut:]


def _write_data_yaml(path: Path) -> None:
    text = "\n".join(
        [
            f"path: {path.parent.resolve()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names: ['ship']",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _stage_one(
    subset: str,
    images: List[Path],
    det_dir: Path,
    out_dir: Path,
    min_conf: float,
    keep_empty: bool,
    max_boxes_per_image: int,
) -> Dict[str, int]:
    img_out = out_dir / "images" / subset
    lbl_out = out_dir / "labels" / subset
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    labeled_images = 0
    kept_boxes = 0
    for img in images:
        det_file = det_dir / f"{img.stem}.json"
        detections = _load_detections(det_file)

        with Image.open(img) as im:
            w, h = im.size

        lines: List[str] = []
        for row in detections:
            conf = _safe_float(row.get("conf"), 0.0)
            if conf < min_conf:
                continue
            xyxy = row.get("xyxy")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                continue
            line = _xyxy_to_yolo_line([float(v) for v in xyxy], w, h, cls_id=0)
            if line:
                lines.append(line)
            if len(lines) >= max_boxes_per_image:
                break

        if not lines and not keep_empty:
            continue

        shutil.copy2(img, img_out / img.name)
        (lbl_out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        kept_images += 1
        kept_boxes += len(lines)
        if lines:
            labeled_images += 1

    return {
        "images": kept_images,
        "labeled_images": labeled_images,
        "boxes": kept_boxes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from S1 images + YOLO detection json pseudo labels")
    parser.add_argument("--image-dir", default="data/interim/s1_grd_png")
    parser.add_argument("--detection-dir", default="outputs/yolo")
    parser.add_argument("--output-dir", default="data/interim/s1_yolo_pseudo")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-conf", type=float, default=0.35)
    parser.add_argument("--max-boxes-per-image", type=int, default=500)
    parser.add_argument("--keep-empty", action="store_true", help="Keep images with empty labels as hard negatives")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    det_dir = Path(args.detection_dir)
    out_dir = Path(args.output_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not det_dir.exists():
        raise FileNotFoundError(f"detection dir not found: {det_dir}")

    imgs = list(_iter_images(image_dir))
    if not imgs:
        raise RuntimeError(f"no images found under {image_dir}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs, val_imgs = _split(imgs, train_ratio=_clip(float(args.train_ratio), 0.1, 0.95), seed=args.seed)
    train_stats = _stage_one(
        subset="train",
        images=train_imgs,
        det_dir=det_dir,
        out_dir=out_dir,
        min_conf=max(0.0, min(1.0, float(args.min_conf))),
        keep_empty=bool(args.keep_empty),
        max_boxes_per_image=max(1, int(args.max_boxes_per_image)),
    )
    val_stats = _stage_one(
        subset="val",
        images=val_imgs,
        det_dir=det_dir,
        out_dir=out_dir,
        min_conf=max(0.0, min(1.0, float(args.min_conf))),
        keep_empty=bool(args.keep_empty),
        max_boxes_per_image=max(1, int(args.max_boxes_per_image)),
    )

    data_yaml = out_dir / "data.yaml"
    _write_data_yaml(data_yaml)

    summary = {
        "image_dir": str(image_dir),
        "detection_dir": str(det_dir),
        "output_dir": str(out_dir),
        "data_yaml": str(data_yaml),
        "total_input_images": len(imgs),
        "train": train_stats,
        "val": val_stats,
        "kept_images": int(train_stats["images"] + val_stats["images"]),
        "labeled_images": int(train_stats["labeled_images"] + val_stats["labeled_images"]),
        "boxes": int(train_stats["boxes"] + val_stats["boxes"]),
        "min_conf": float(args.min_conf),
        "keep_empty": bool(args.keep_empty),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[done] dataset="
        f"{out_dir} kept_images={summary['kept_images']} labeled_images={summary['labeled_images']} boxes={summary['boxes']}"
    )
    print(f"[done] data_yaml={data_yaml}")


if __name__ == "__main__":
    main()

