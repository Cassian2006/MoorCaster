from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _count_files(folder: Path, exts: set[str]) -> int:
    if not folder.exists():
        return 0
    n = 0
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.name == ".gitkeep":
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        n += 1
    return n


def _is_ais_download_running() -> bool:
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*download_ais_pipeline.py*' -or $_.CommandLine -like '*download_ais_complete.py*' } | "
        "Measure-Object | Select-Object -ExpandProperty Count"
    )
    p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, cwd=str(ROOT))
    if p.returncode != 0:
        return False
    try:
        return int((p.stdout or "0").strip()) > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for AIS/S1 readiness then run full pipeline automatically")
    parser.add_argument("--ais-dir", default="data/raw/ais")
    parser.add_argument("--s1-dir", default="data/raw/s1/quicklook")
    parser.add_argument("--ais-min-files", type=int, default=1)
    parser.add_argument("--s1-min-files", type=int, default=5)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--max-wait-min", type=int, default=24 * 60)
    parser.add_argument("--horizon-days", type=int, default=24)
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--start-now-if-ais-running", action="store_true")
    parser.add_argument("--allow-ais-running", action="store_true", help="Do not require AIS downloader to stop")
    args = parser.parse_args()

    ais_dir = ROOT / args.ais_dir
    s1_dir = ROOT / args.s1_dir

    deadline = time.time() + max(1, args.max_wait_min) * 60
    while True:
        ais_count = _count_files(ais_dir, {".csv", ".jsonl"})
        s1_count = _count_files(s1_dir, {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        ais_running = _is_ais_download_running()
        print(f"[wait] ais_count={ais_count} s1_count={s1_count} ais_running={ais_running}", flush=True)

        ready_by_count = ais_count >= args.ais_min_files and s1_count >= args.s1_min_files
        ready_by_running = args.start_now_if_ais_running and ais_running and s1_count >= args.s1_min_files
        if (not args.allow_ais_running) and ais_running:
            ready_by_count = False

        if ready_by_count or ready_by_running:
            break
        if time.time() >= deadline:
            raise TimeoutError("wait timeout: data not ready")
        time.sleep(max(5, args.poll_sec))

    # Ensure S1 quicklooks are synced into YOLO input folder first.
    subprocess.run(
        [
            sys.executable,
            "scripts/sync_s1_quicklook.py",
            "--src",
            "data/raw/s1/quicklook",
            "--dst",
            "data/interim/s1_quicklook",
            "--mode",
            "copy",
        ],
        cwd=str(ROOT),
        check=False,
    )

    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--horizon-days",
        str(args.horizon_days),
    ]
    if args.yolo_model.strip():
        cmd += ["--yolo-model", args.yolo_model.strip()]
    print(f"[run] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError("run_pipeline failed")
    print("[done] pipeline completed after wait", flush=True)


if __name__ == "__main__":
    main()
