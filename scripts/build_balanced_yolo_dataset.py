from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _is_positive(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    txt = label_path.read_text(encoding="utf-8").strip()
    return bool(txt)


def _pairs(src_root: Path, split: str) -> List[Tuple[Path, Path]]:
    img_dir = src_root / "images" / split
    lbl_dir = src_root / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"missing images/labels under {src_root} for split={split}")

    out: List[Tuple[Path, Path]] = []
    for p in sorted(img_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            out.append((p, lbl))
    if not out:
        raise RuntimeError(f"no image-label pairs found in {split_dir}")
    return out


def _copy_pairs(pairs: List[Tuple[Path, Path]], out_dir: Path, split: str) -> Dict[str, int]:
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)
        txt = lbl.read_text(encoding="utf-8").strip()
        boxes += 0 if not txt else len([ln for ln in txt.splitlines() if ln.strip()])
    return {"images": len(pairs), "labels": len(pairs), "boxes": boxes}


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
    parser = argparse.ArgumentParser(description="Build balanced YOLO dataset by downsampling empty-label negatives.")
    parser.add_argument("--src", default="data/interim/lssdd_yolo")
    parser.add_argument("--out", default="data/interim/lssdd_yolo_balanced")
    parser.add_argument("--neg-pos-ratio", type=float, default=1.0, help="Keep at most ratio * positive negatives in train.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-val-test", action="store_true", help="Also copy val/test splits unchanged.")
    args = parser.parse_args()

    src_dir = ROOT / args.src if not Path(args.src).is_absolute() else Path(args.src)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    if args.copy_val_test:
        (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

    train_pairs = _pairs(src_dir, "train")
    pos = [(img, lbl) for img, lbl in train_pairs if _is_positive(lbl)]
    neg = [(img, lbl) for img, lbl in train_pairs if not _is_positive(lbl)]

    keep_neg = int(round(max(0.0, float(args.neg_pos_ratio)) * len(pos)))
    keep_neg = min(keep_neg, len(neg))
    rnd = random.Random(args.seed)
    neg_sel = rnd.sample(neg, keep_neg) if keep_neg > 0 else []
    train_bal = pos + neg_sel
    rnd.shuffle(train_bal)

    train_stats = _copy_pairs(train_bal, out_dir, "train")
    val_stats = {"images": 0, "labels": 0, "boxes": 0}
    test_stats = {"images": 0, "labels": 0, "boxes": 0}
    if args.copy_val_test:
        val_stats = _copy_pairs(_pairs(src_dir, "val"), out_dir, "val")
        test_stats = _copy_pairs(_pairs(src_dir, "test"), out_dir, "test")

    _write_data_yaml(out_dir)

    summary = {
        "src": str(src_dir.resolve()),
        "out": str(out_dir.resolve()),
        "neg_pos_ratio": float(args.neg_pos_ratio),
        "seed": int(args.seed),
        "copy_val_test": bool(args.copy_val_test),
        "train_src_total": len(train_pairs),
        "train_src_pos": len(pos),
        "train_src_neg": len(neg),
        "train_kept_pos": len(pos),
        "train_kept_neg": len(neg_sel),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] balanced data yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
