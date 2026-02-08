import argparse
from pathlib import Path
import sys

# Ensure project root is importable when running `python scripts/...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.yolo.infer import load_model, run_inference, run_inference_tiled, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a folder of images")
    parser.add_argument("--model", required=True, help="Path to YOLO weights")
    parser.add_argument("--input-dir", "--input", dest="input_dir", required=True, help="Directory with images")
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True, help="Directory to save detections")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class ids to keep, e.g. '8' for COCO boat, '52' for OIV7 Boat",
    )
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference for large satellite images")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()

    model = load_model(args.model)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = []
    if args.classes.strip():
        classes = [int(x.strip()) for x in args.classes.split(",") if x.strip() != ""]

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
    for img in images:
        if args.tiled:
            detections = run_inference_tiled(
                model,
                img,
                conf=args.conf,
                iou=args.iou,
                classes=classes or None,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                nms_iou=args.nms_iou,
            )
        else:
            detections = run_inference(model, img, conf=args.conf, iou=args.iou, classes=classes or None)
        save_json(detections, output_dir / f"{img.stem}.json")
        print(f"{img.name}: {len(detections)} detections")


if __name__ == "__main__":
    main()
