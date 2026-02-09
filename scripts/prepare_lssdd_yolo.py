from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _read_ids(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"missing split file: {txt_path}")
    return [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_annotations_dir(ls_root: Path) -> Path:
    candidates = [
        ls_root / "Annotations_sub" / "Annotations_sub",
        ls_root / "Annotations_sub",
        ls_root / "Annotations",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("annotation directory not found under LS-SSDD root")


def _find_images(ls_root: Path) -> Dict[str, Path]:
    img_map: Dict[str, Path] = {}
    for p in ls_root.rglob("*.jpg"):
        img_map[p.stem] = p
    if not img_map:
        raise FileNotFoundError("no .jpg images found under LS-SSDD root")
    return img_map


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _xml_to_yolo_lines(xml_path: Path) -> Tuple[int, int, List[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.findtext("size/width", default="0"))
    height = int(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in xml: {xml_path}")

    lines: List[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name", default="ship") or "ship").strip().lower()
        if name and name != "ship":
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue

        xmin = float(bb.findtext("xmin", default="0"))
        ymin = float(bb.findtext("ymin", default="0"))
        xmax = float(bb.findtext("xmax", default="0"))
        ymax = float(bb.findtext("ymax", default="0"))

        xmin = _clamp(xmin, 0.0, float(width))
        xmax = _clamp(xmax, 0.0, float(width))
        ymin = _clamp(ymin, 0.0, float(height))
        ymax = _clamp(ymax, 0.0, float(height))
        if xmax <= xmin or ymax <= ymin:
            continue

        x = ((xmin + xmax) * 0.5) / width
        y = ((ymin + ymax) * 0.5) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return width, height, lines


def _ensure_clean_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path) -> None:
    payload = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [ship]\n"
    )
    (out_dir / "data.yaml").write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LS-SSDDv1 into YOLO format with clean splits")
    parser.add_argument("--ls-root", default=r"C:\Users\cai yuan qi\Desktop\LS-SSDDv1")
    parser.add_argument("--out-dir", default="data/interim/lssdd_yolo")
    parser.add_argument(
        "--drop-overlap",
        action="store_true",
        help="Drop train/val/test overlap by priority test > val > train (recommended)",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images without ship boxes as empty-label negatives",
    )
    args = parser.parse_args()

    ls_root = Path(args.ls_root)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    ann_dir = _find_annotations_dir(ls_root)
    image_map = _find_images(ls_root)

    train_ids = _read_ids(ls_root / "train.txt")
    val_ids = _read_ids(ls_root / "val.txt")
    test_ids = _read_ids(ls_root / "test.txt")
    original = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    overlap_before = {
        "train_val": len(train_set & val_set),
        "train_test": len(train_set & test_set),
        "val_test": len(val_set & test_set),
    }

    if args.drop_overlap:
        val_set = val_set - test_set
        train_set = train_set - val_set - test_set

    splits = {
        "train": sorted(train_set),
        "val": sorted(val_set),
        "test": sorted(test_set),
    }
    overlap_after = {
        "train_val": len(set(splits["train"]) & set(splits["val"])),
        "train_test": len(set(splits["train"]) & set(splits["test"])),
        "val_test": len(set(splits["val"]) & set(splits["test"])),
    }

    _ensure_clean_dirs(
        [
            out_dir / "images" / "train",
            out_dir / "images" / "val",
            out_dir / "images" / "test",
            out_dir / "labels" / "train",
            out_dir / "labels" / "val",
            out_dir / "labels" / "test",
        ]
    )

    stats: Dict[str, Dict[str, int]] = {}
    for split, ids in splits.items():
        st = {
            "requested": len(ids),
            "copied_images": 0,
            "written_labels": 0,
            "boxes": 0,
            "missing_image": 0,
            "missing_xml": 0,
            "empty_label_kept": 0,
            "empty_label_skipped": 0,
        }
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split

        for sid in ids:
            xml_path = ann_dir / f"{sid}.xml"
            if not xml_path.exists():
                st["missing_xml"] += 1
                continue
            img_path = image_map.get(sid)
            if img_path is None or not img_path.exists():
                st["missing_image"] += 1
                continue

            _, _, lines = _xml_to_yolo_lines(xml_path)
            if not lines and not args.keep_empty:
                st["empty_label_skipped"] += 1
                continue

            shutil.copy2(img_path, img_out / img_path.name)
            (lbl_out / f"{sid}.txt").write_text("\n".join(lines), encoding="utf-8")
            st["copied_images"] += 1
            st["written_labels"] += 1
            st["boxes"] += len(lines)
            if not lines:
                st["empty_label_kept"] += 1

        stats[split] = st

    _write_data_yaml(out_dir)
    summary = {
        "ls_root": str(ls_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "drop_overlap": bool(args.drop_overlap),
        "keep_empty": bool(args.keep_empty),
        "original_splits": original,
        "overlap_before": overlap_before,
        "overlap_after": overlap_after,
        "prepared_splits": {k: len(v) for k, v in splits.items()},
        "stats": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] data yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
