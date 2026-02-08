from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_build_s1_pseudo_yolo_dataset(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    det_dir = tmp_path / "detections"
    out_dir = tmp_path / "dataset"
    img_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    img_a = img_dir / "a.png"
    img_b = img_dir / "b.png"
    Image.new("RGB", (100, 80), color=(0, 0, 0)).save(img_a)
    Image.new("RGB", (120, 120), color=(0, 0, 0)).save(img_b)

    (det_dir / "a.json").write_text(
        json.dumps(
            [
                {"xyxy": [10, 10, 50, 40], "cls": 8, "conf": 0.9},
                {"xyxy": [1, 1, 2, 2], "cls": 8, "conf": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    (det_dir / "b.json").write_text("[]", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/build_s1_pseudo_yolo_dataset.py",
        "--image-dir",
        str(img_dir),
        "--detection-dir",
        str(det_dir),
        "--output-dir",
        str(out_dir),
        "--min-conf",
        "0.35",
        "--train-ratio",
        "0.5",
        "--seed",
        "7",
        "--keep-empty",
        "--clean",
    ]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    data_yaml = out_dir / "data.yaml"
    summary_json = out_dir / "summary.json"
    assert data_yaml.exists()
    assert summary_json.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["total_input_images"] == 2
    assert summary["kept_images"] == 2
    assert summary["labeled_images"] == 1
    assert summary["boxes"] == 1

    train_imgs = list((out_dir / "images" / "train").glob("*.png"))
    val_imgs = list((out_dir / "images" / "val").glob("*.png"))
    assert len(train_imgs) + len(val_imgs) == 2

    label_files = list((out_dir / "labels" / "train").glob("*.txt")) + list((out_dir / "labels" / "val").glob("*.txt"))
    assert len(label_files) == 2
    assert any(fp.read_text(encoding="utf-8").strip() for fp in label_files)
