from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    full = [sys.executable] + cmd
    print(f"[run] {' '.join(full)}", flush=True)
    p = subprocess.run(full, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def _log_has_done(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "[done] downloaded_new=" in text


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait S1 product download completion then run prepare+yolo")
    parser.add_argument("--download-log", default="outputs/logs/download_s1_product.log")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    log_path = ROOT / args.download_log
    print(f"[wait] watching {log_path}", flush=True)

    while True:
        if _log_has_done(log_path):
            break
        print("[wait] S1 download not finished yet...", flush=True)
        time.sleep(max(5, args.poll_sec))

    print("[wait] S1 download done. start prepare_s1_grd + run_s1_yolo_auto", flush=True)
    _run(
        [
            "scripts/prepare_s1_grd.py",
            "--input-dir",
            "data/raw/s1/grd_zip",
            "--output-dir",
            "data/interim/s1_grd_png",
            "--prefer-pol",
            "vv",
        ]
    )
    _run(["scripts/run_s1_yolo_auto.py", "--horizon", str(args.horizon)])
    print("[done] wait_s1_and_run_yolo completed", flush=True)


if __name__ == "__main__":
    main()
