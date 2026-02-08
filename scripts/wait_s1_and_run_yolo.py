from __future__ import annotations

import argparse
import re
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


_DONE_RE = re.compile(r"^\[download\] done (?P<name>.+?\.zip)\s", re.MULTILINE)


def _done_files_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    names = set()
    for m in _DONE_RE.finditer(text):
        names.add(m.group("name"))
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait S1 product download completion then run prepare+yolo")
    parser.add_argument("--download-log", default="outputs/logs/download_s1_product.log")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--required-done-count", type=int, default=8)
    args = parser.parse_args()

    log_path = ROOT / args.download_log
    print(f"[wait] watching {log_path}", flush=True)

    while True:
        done_count = _done_files_count(log_path)
        print(f"[wait] S1 completed files: {done_count}/{args.required_done_count}", flush=True)
        if done_count >= args.required_done_count:
            break
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
