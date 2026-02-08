import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _parse_time_from_stem(stem: str) -> Optional[datetime]:
    patterns = [
        (r"(20\d{2})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", "%Y%m%d%H%M%S"),
        (r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[T _-]?(\d{2})[:_-]?(\d{2})[:_-]?(\d{2})", "%Y%m%d%H%M%S"),
        (r"(20\d{2})(\d{2})(\d{2})", "%Y%m%d"),
        (r"(20\d{2})[-_](\d{2})[-_](\d{2})", "%Y%m%d"),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, stem)
        if not m:
            continue
        try:
            text = "".join(m.groups())
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _read_detections(path: Path) -> List[Dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [d for d in payload if isinstance(d, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("detections"), list):
            return [d for d in payload["detections"] if isinstance(d, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO observed timeseries from detection json files")
    parser.add_argument("--yolo-dir", required=True, help="Directory with YOLO json outputs")
    parser.add_argument("--output", default="outputs/metrics/yolo_observed.csv")
    parser.add_argument("--time-bin", default="1D", help="Resample/floor frequency (default: 1D)")
    args = parser.parse_args()

    yolo_dir = Path(args.yolo_dir)
    if not yolo_dir.exists():
        raise FileNotFoundError(f"yolo dir not found: {yolo_dir}")

    rows: List[Dict] = []
    skipped_time = 0
    for fp in sorted(yolo_dir.glob("*.json")):
        ts = _parse_time_from_stem(fp.stem)
        if ts is None:
            skipped_time += 1
            continue
        detections = _read_detections(fp)
        conf_vals = [float(d["conf"]) for d in detections if isinstance(d.get("conf"), (int, float))]
        rows.append(
            {
                "timestamp": ts,
                "file": fp.name,
                "detections": len(detections),
                "mean_conf": (sum(conf_vals) / len(conf_vals)) if conf_vals else None,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame(columns=["time_bin", "yolo_detections", "yolo_files", "yolo_mean_conf"]).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        print(f"yolo observed -> {out_path} (empty, skipped_time={skipped_time})")
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_bin"] = df["timestamp"].dt.floor(args.time_bin)

    # Confidence is weighted by number of detections per file for a stable daily metric.
    df["weighted_conf"] = df["mean_conf"] * df["detections"]
    grouped = (
        df.groupby("time_bin", as_index=False)
        .agg(
            yolo_detections=("detections", "sum"),
            yolo_files=("file", "count"),
            conf_weight_sum=("weighted_conf", "sum"),
            conf_weight_n=("detections", "sum"),
        )
        .sort_values("time_bin")
    )
    grouped["yolo_mean_conf"] = grouped.apply(
        lambda r: (r["conf_weight_sum"] / r["conf_weight_n"]) if r["conf_weight_n"] > 0 else None,
        axis=1,
    )
    out = grouped[["time_bin", "yolo_detections", "yolo_files", "yolo_mean_conf"]]
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"yolo observed -> {out_path} bins={len(out)} files={len(df)} skipped_time={skipped_time}"
    )


if __name__ == "__main__":
    main()

