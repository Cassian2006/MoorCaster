import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_time(text: str):
    cleaned = str(text).replace("T", " ")
    return cleaned


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


def collect_points(input_dir: Path, raw_dir: str, raw_csv_dir: str) -> List[Dict]:
    points: List[Dict] = []
    jsonl_root = input_dir / raw_dir
    if jsonl_root.exists():
        for file in jsonl_root.rglob("*.jsonl"):
            points.extend(list(read_jsonl(file)))
    csv_root = input_dir / raw_csv_dir
    if csv_root.exists():
        for file in csv_root.rglob("*.csv"):
            points.extend(list(read_csv_points(file)))
    return points


def dedupe_points(points: Iterable[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for p in points:
        key = (str(p.get("mmsi")), str(p.get("postime")))
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    result.sort(key=lambda x: (str(x.get("mmsi")), parse_time(x.get("postime"))))
    return result


def save_csv(points: Iterable[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(points)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge AIS slice files (jsonl or csv) into one CSV with dedupe.")
    parser.add_argument("--input-dir", required=True, help="Output directory containing raw slices.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--raw-dir", default="raw_tracks", help="Subdir name for jsonl slices.")
    parser.add_argument("--raw-csv-dir", default="raw_tracks_csv", help="Subdir name for csv slices.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_points = collect_points(input_dir, args.raw_dir, args.raw_csv_dir)
    merged = dedupe_points(all_points)
    save_csv(merged, Path(args.output))
    print(f"Merged {len(all_points)} points, {len(merged)} after dedupe into {args.output}")


if __name__ == "__main__":
    main()
