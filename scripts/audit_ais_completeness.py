import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_time(text: str):
    return str(text).replace("T", " ")


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_csv_points(path: Path) -> Iterable[Dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_mmsi_csv(path: Path) -> List[str]:
    result = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if not row:
                continue
            value = row[0].strip()
            if idx == 0 and value.lower() == "mmsi":
                continue
            if not value.isdigit():
                continue
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def build_slices(start: str, end: str, slice_hours: int) -> List[Tuple[str, str, str]]:
    import datetime as dt

    start_dt = dt.datetime.fromisoformat(start.replace("T", " "))
    end_dt = dt.datetime.fromisoformat(end.replace("T", " "))
    delta = dt.timedelta(hours=slice_hours)
    cursor = start_dt
    slices = []
    while cursor < end_dt:
        nxt = min(end_dt, cursor + delta)
        token = f"{cursor.strftime('%Y%m%dT%H%M%S')}__{nxt.strftime('%Y%m%dT%H%M%S')}"
        slices.append((token, cursor, nxt))
        cursor = nxt
    return slices


def audit(output_dir: Path, raw_dir: str, raw_csv_dir: str, start: str, end: str, slice_hours: int):
    mmsi_list = load_mmsi_csv(output_dir / "mmsi.csv")
    slices = build_slices(start, end, slice_hours)
    coverage_rows = []
    slice_counts = defaultdict(int)
    total_points = 0

    for mmsi in mmsi_list:
        existing = 0
        missing_tokens = []
        mmsi_points = 0
        for token, _, _ in slices:
            csv_path = output_dir / raw_csv_dir / mmsi / f"{token}.csv"
            jsonl_path = output_dir / raw_dir / mmsi / f"{token}.jsonl"
            points = []
            if csv_path.exists():
                points = list(read_csv_points(csv_path))
            elif jsonl_path.exists():
                points = list(read_jsonl(jsonl_path))
            if points:
                existing += 1
                mmsi_points += len(points)
                slice_counts[token] += len(points)
                total_points += len(points)
            else:
                missing_tokens.append(token)
        coverage_rows.append(
            {
                "mmsi": mmsi,
                "expected_slices": len(slices),
                "existing_slices": existing,
                "missing_slices": len(missing_tokens),
                "missing_tokens": ";".join(missing_tokens),
                "total_points": mmsi_points,
            }
        )

    cov_path = output_dir / "coverage_report.csv"
    fieldnames = ["mmsi", "expected_slices", "existing_slices", "missing_slices", "missing_tokens", "total_points"]
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    with cov_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    summary_path = output_dir / "audit_summary.md"
    failed_path = output_dir / "failed_windows.csv"
    failed_cnt = 0
    if failed_path.exists():
        with failed_path.open(encoding="utf-8") as f:
            failed_cnt = max(0, sum(1 for _ in f) - 1)

    top_slices = sorted(slice_counts.items(), key=lambda x: x[0])
    summary_lines = [
        f"# AIS Audit Summary",
        f"- total mmsi: {len(mmsi_list)}",
        f"- total points: {total_points}",
        f"- failed_windows rows: {failed_cnt}",
        f"- coverage_report: {cov_path.name}",
        "",
        "## Points by slice (token => points)",
    ]
    for token, count in top_slices:
        summary_lines.append(f"- {token}: {count}")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Audit complete. Coverage -> {cov_path}, summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit AIS completeness from slices.")
    parser.add_argument("--output-dir", required=True, help="Output directory from downloader.")
    parser.add_argument("--start", required=True, help="Start time used for download.")
    parser.add_argument("--end", required=True, help="End time used for download.")
    parser.add_argument("--slice-hours", type=int, default=6, help="Slice hours used for download.")
    parser.add_argument("--raw-dir", default="raw_tracks", help="Subdir for jsonl slices.")
    parser.add_argument("--raw-csv-dir", default="raw_tracks_csv", help="Subdir for csv slices.")
    args = parser.parse_args()

    audit(Path(args.output_dir), args.raw_dir, args.raw_csv_dir, args.start, args.end, args.slice_hours)


if __name__ == "__main__":
    main()
