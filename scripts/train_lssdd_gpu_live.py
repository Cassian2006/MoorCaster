from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _clean_line(s: str) -> str:
    s = ANSI_RE.sub("", s)
    s = s.replace("\x00", "").replace("\x08", "")
    s = " ".join(s.split())
    return s.strip()


def _short(s: str, n: int = 160) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _build_cmd(args: argparse.Namespace) -> list[str]:
    base = args.base_model.strip()
    if not base:
        candidate = ROOT / "assets" / "models" / "moorcaster_ship_lssdd_smoke.pt"
        base = str(candidate) if candidate.exists() else "yolov8n.pt"

    return [
        sys.executable,
        "scripts/train_s1_yolo_finetune.py",
        "--data-yaml",
        args.data_yaml,
        "--base-model",
        base,
        "--epochs",
        str(max(1, int(args.epochs))),
        "--imgsz",
        str(max(320, int(args.imgsz))),
        "--batch",
        str(max(1, int(args.batch))),
        "--device",
        args.device,
        "--workers",
        str(max(0, int(args.workers))),
        "--patience",
        str(max(1, int(args.patience))),
        "--project",
        args.project,
        "--run-name",
        args.run_name,
        "--export-model",
        args.export_model,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LS-SSDD YOLO training on GPU and print readable live progress."
    )
    parser.add_argument("--data-yaml", default="data/interim/lssdd_yolo/data.yaml")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU id, e.g. 0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", default="outputs/train")
    parser.add_argument("--run-name", default="lssdd_full_30ep_gpu")
    parser.add_argument("--export-model", default="assets/models/moorcaster_ship_lssdd.pt")
    parser.add_argument("--log-file", default="outputs/logs/train_lssdd_gpu_live.log")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    cmd = _build_cmd(args)
    log_path = ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print("[start] launching training command:", flush=True)
    print(" ".join(cmd), flush=True)
    print(f"[log] {log_path}", flush=True)

    started = time.time()
    last_heartbeat = started
    last_line = ""

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            now = time.time()

            if now - last_heartbeat >= max(5, int(args.heartbeat_sec)):
                elapsed_min = (now - started) / 60.0
                msg = f"[heartbeat] elapsed={elapsed_min:.1f} min, last={_short(last_line)}"
                print(msg, flush=True)
                lf.write(msg + "\n")
                lf.flush()
                last_heartbeat = now

            if ch == "":
                if buf:
                    line = _clean_line(buf)
                    if line:
                        print(line, flush=True)
                        lf.write(line + "\n")
                        last_line = line
                    buf = ""
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            if ch in ("\r", "\n"):
                if buf:
                    line = _clean_line(buf)
                    if line and line != last_line:
                        print(line, flush=True)
                        lf.write(line + "\n")
                        last_line = line
                    buf = ""
                continue

            buf += ch

        rc = proc.wait()
        elapsed = (time.time() - started) / 60.0
        tail = f"[done] return_code={rc} elapsed={elapsed:.1f} min log={log_path}"
        print(tail, flush=True)
        lf.write(tail + "\n")
        lf.write(f"===== ended {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lf.flush()
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
