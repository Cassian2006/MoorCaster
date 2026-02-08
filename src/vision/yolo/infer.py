import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install it before running YOLO inference."
        ) from exc
    return YOLO


def load_model(model_path: str):
    YOLO = _require_ultralytics()
    return YOLO(model_path)


def run_inference(
    model,
    image_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    kwargs = {"source": str(image_path), "conf": conf, "iou": iou, "verbose": False}
    if classes:
        kwargs["classes"] = classes
    results = model.predict(**kwargs)
    outputs: List[Dict[str, Any]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            xyxy = b.xyxy[0].tolist() if hasattr(b, "xyxy") else None
            cls = int(b.cls[0]) if hasattr(b, "cls") else None
            conf_val = float(b.conf[0]) if hasattr(b, "conf") else None
            outputs.append({"xyxy": xyxy, "cls": cls, "conf": conf_val})
    return outputs


def _box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _nms(records: List[Dict[str, Any]], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    if not records:
        return records
    order = sorted(records, key=lambda x: float(x.get("conf") or 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []
    for r in order:
        box = r.get("xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        cls = r.get("cls")
        suppress = False
        for k in kept:
            if cls != k.get("cls"):
                continue
            kbox = k.get("xyxy")
            if not isinstance(kbox, list) or len(kbox) != 4:
                continue
            if _box_iou_xyxy(box, kbox) >= iou_thr:
                suppress = True
                break
        if not suppress:
            kept.append(r)
    return kept


def _iter_tiles(width: int, height: int, tile_size: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    tile = max(128, int(tile_size))
    ov = min(max(float(overlap), 0.0), 0.9)
    step = max(32, int(tile * (1.0 - ov)))
    xs = list(range(0, max(1, width - tile + 1), step))
    ys = list(range(0, max(1, height - tile + 1), step))
    if not xs or xs[-1] != max(0, width - tile):
        xs.append(max(0, width - tile))
    if not ys or ys[-1] != max(0, height - tile):
        ys.append(max(0, height - tile))

    tiles: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x2 = min(width, x + tile)
            y2 = min(height, y + tile)
            tiles.append((x, y, x2, y2))
    return tiles


def run_inference_tiled(
    model,
    image_path: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    classes: Optional[List[int]] = None,
    tile_size: int = 1024,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
) -> List[Dict[str, Any]]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    tiles = _iter_tiles(width=w, height=h, tile_size=tile_size, overlap=overlap)

    outputs: List[Dict[str, Any]] = []
    for x1, y1, x2, y2 in tiles:
        tile = arr[y1:y2, x1:x2]
        kwargs = {"source": tile, "conf": conf, "iou": iou, "verbose": False}
        if classes:
            kwargs["classes"] = classes
        results = model.predict(**kwargs)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                xyxy = b.xyxy[0].tolist() if hasattr(b, "xyxy") else None
                if not xyxy:
                    continue
                xyxy = [xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1]
                cls = int(b.cls[0]) if hasattr(b, "cls") else None
                conf_val = float(b.conf[0]) if hasattr(b, "conf") else None
                outputs.append({"xyxy": xyxy, "cls": cls, "conf": conf_val})

    return _nms(outputs, iou_thr=nms_iou)


def save_json(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
