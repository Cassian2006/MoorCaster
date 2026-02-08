from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import List


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "raw" / "ais"
DEFAULT_START = "2024-04-01 00:00:00"
DEFAULT_END = "2024-06-30 23:59:59"
DEFAULT_BBOX = "121.90,30.75,122.25,30.50"


def _scan_progress(out_dir: Path) -> tuple[int, str, str]:
    files: List[Path] = []
    for sub in ("raw_tracks_csv", "raw_tracks"):
        base = out_dir / sub
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            # Ignore in-flight temp files; they may be moved/removed during scan.
            if p.suffix.lower() == ".tmp":
                continue
            # Only count final slice artifacts.
            if p.suffix.lower() not in {".csv", ".jsonl"}:
                continue
            files.append(p)
    if not files:
        return 0, "", ""

    stamped: List[tuple[float, Path]] = []
    for p in files:
        try:
            stamped.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            # File may be atomically replaced while scanning.
            continue
    if not stamped:
        return 0, "", ""
    stamped.sort(key=lambda x: x[0], reverse=True)
    latest_mtime, latest = stamped[0]
    latest_time = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return len(stamped), str(latest), latest_time


def _enqueue_stdout(pipe, q: Queue[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            q.put(line.rstrip("\n"))
    finally:
        pipe.close()


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (os.getenv("MYVESSEL_CLIENT_ID") and os.getenv("MYVESSEL_CLIENT_SECRET")):
        print("[error] missing MYVESSEL_CLIENT_ID / MYVESSEL_CLIENT_SECRET")
        return 2

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "download_ais_complete.py"),
        "--start",
        args.start,
        "--end",
        args.end,
        "--region",
        args.bbox,
        "--output-dir",
        str(out_dir),
        "--page-size",
        str(args.page_size),
        "--slice-hours",
        str(args.slice_hours),
        "--slice-threshold",
        str(args.slice_threshold),
        "--output-format",
        args.output_format,
    ]

    print("[init] starting AIS download")
    print(f"[init] bbox={args.bbox} start={args.start} end={args.end}")
    print(f"[init] output={out_dir}")
    print(f"[init] cmd={' '.join(cmd)}")
    print("[resume] existing slice files will be skipped automatically")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    q: Queue[str] = Queue()
    t = threading.Thread(target=_enqueue_stdout, args=(proc.stdout, q), daemon=True)
    t.start()

    last_count = -1
    last_latest = ""
    last_tick = 0.0
    progress_interval = max(1.0, float(args.progress_interval))

    while proc.poll() is None:
        now = time.time()
        if now - last_tick >= progress_interval:
            count, latest, latest_time = _scan_progress(out_dir)
            if count != last_count:
                print(f"[progress] files={count} latest={latest_time} {latest}")
                last_count = count
                last_latest = latest
            elif latest and latest != last_latest:
                print(f"[progress] latest switched -> {latest_time} {latest}")
                last_latest = latest
            last_tick = now

        try:
            line = q.get(timeout=0.2).strip()
            if line:
                print(f"[downloader] {line}")
        except Empty:
            pass

    # Drain remaining output
    while True:
        try:
            line = q.get_nowait().strip()
            if line:
                print(f"[downloader] {line}")
        except Empty:
            break

    rc = proc.returncode or 0
    count, latest, latest_time = _scan_progress(out_dir)
    if rc == 0:
        print(f"[done] rc=0 files={count} latest={latest_time} {latest}")
    else:
        print(f"[error] rc={rc} files={count} latest={latest_time} {latest}")
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Main AIS download entry with resume and progress prints")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--bbox", default=DEFAULT_BBOX, help="lon1,lat1,lon2,lat2")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--slice-hours", type=int, default=6)
    p.add_argument("--slice-threshold", type=int, default=9000)
    p.add_argument("--output-format", choices=["csv", "jsonl", "both"], default="both")
    p.add_argument("--progress-interval", type=float, default=3.0, help="seconds")
    return p


def main() -> None:
    args = build_parser().parse_args()
    rc = run(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
