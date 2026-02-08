from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.congestion_waiting import MetricConfig, run_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build congestion and waiting metrics from cleaned AIS")
    parser.add_argument("--clean-dir", default="data/processed/ais_cleaned")
    parser.add_argument("--out-dir", default="outputs/metrics")
    parser.add_argument("--time-bin", default="1H")
    parser.add_argument("--sog-threshold", type=float, default=0.5)
    parser.add_argument("--max-gap-min", type=int, default=60)
    parser.add_argument("--min-duration-min", type=int, default=20)
    args = parser.parse_args()

    cfg = MetricConfig(
        time_bin=args.time_bin,
        sog_threshold=args.sog_threshold,
        max_gap_min=args.max_gap_min,
        min_duration_min=args.min_duration_min,
    )
    run_metrics(
        clean_dir=Path(args.clean_dir),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    print(f"metrics -> {args.out_dir}")


if __name__ == "__main__":
    main()

