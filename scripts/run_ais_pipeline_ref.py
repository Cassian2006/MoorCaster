import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

# Reuse existing modules
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import Ais_downloader.download_ais_complete as dl

TOKEN_URL = "https://svc.data.myvessel.cn/ada/oauth/token"


def get_token(client_id: str, client_secret: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        params={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def normalize_bbox(lon1, lat1, lon2, lat2):
    if lon1 > lon2:
        lon1, lon2 = lon2, lon1
    if lat1 < lat2:
        lat1, lat2 = lat2, lat1
    return lon1, lat1, lon2, lat2


def tile_bbox(bbox: str, tile_deg: float) -> List[str]:
    lon1, lat1, lon2, lat2 = [float(x) for x in bbox.split(",")]
    lon1, lat1, lon2, lat2 = normalize_bbox(lon1, lat1, lon2, lat2)
    tiles = []
    lat = lat1
    while lat > lat2:
        next_lat = max(lat2, lat - tile_deg)
        lon = lon1
        while lon < lon2:
            next_lon = min(lon2, lon + tile_deg)
            tiles.append(f"{lon},{lat},{next_lon},{next_lat}")
            lon = next_lon
        lat = next_lat
    return tiles


def census_mmsi(start: dt.datetime, end: dt.datetime, bboxes: List[str], tile_deg: float, page_size: int, client_id: str, client_secret: str, out_dir: Path):
    token = get_token(client_id, client_secret)
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "ais-pipeline/census"}
    session = requests.Session()
    mmsi_set = set()
    stats_rows = []
    ensure_dir(out_dir)
    for bbox in bboxes:
        tiles = tile_bbox(bbox, tile_deg)
        for idx, tile in enumerate(tiles):
            marker = out_dir / f".census_done_{tile.replace(',','_')}"
            if marker.exists():
                continue
            region = dl.region_from_bbox(tile)
            page_num = 1
            total_elements = 0
            pages = 0
            while True:
                payload = {
                    "dwt": 0,
                    "teu": 0,
                    "grt": 0,
                    "vesselSubType": [],
                    "teu2": 40000,
                    "grt2": 40000,
                    "startTime": dl.isoformat(start),
                    "endTime": dl.isoformat(end),
                    "page": {"pageSize": page_size, "pageNum": page_num},
                    "region": region,
                    "dwt2": 400000,
                }
                resp = session.post(dl.REGION_EVENTS_URL, headers=headers, json=payload, timeout=20)
                body = resp.json()
                data = body.get("data") or {}
                content = dl.extract_list_from_data(data)
                for r in content:
                    val = r.get("mmsi")
                    if val and str(val).isdigit():
                        mmsi_set.add(str(val))
                total_elements += len(content)
                pages += 1
                total_pages = data.get("totalPages") if isinstance(data, dict) else None
                if not content or len(content) < page_size or (total_pages and page_num >= total_pages):
                    break
                page_num += 1
            stats_rows.append({"tile": tile, "pages": pages, "records": total_elements})
            marker.write_text("done", encoding="utf-8")
    mmsi_path = out_dir / "mmsi_registry.csv"
    with mmsi_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mmsi"])
        for m in sorted(mmsi_set):
            writer.writerow([m])
    # also write mmsi.csv for downstream tools
    with (out_dir / "mmsi.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mmsi"])
        for m in sorted(mmsi_set):
            writer.writerow([m])
    census_path = out_dir / "census_stats.csv"
    with census_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["tile", "pages", "records"])
        writer.writeheader()
        writer.writerows(stats_rows)
    return mmsi_path, census_path


def clip_csv(src: Path, bboxes: List[str], dst: Path):
    if not src.exists():
        return 0, 0
    df = pd.read_csv(src)
    if df.empty:
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        return 0, 0
    keep = []
    for bbox in bboxes:
        lon1, lat1, lon2, lat2 = normalize_bbox(*[float(x) for x in bbox.split(",")])
        mask = (df["lon"] >= lon1) & (df["lon"] <= lon2) & (df["lat"] <= lat1) & (df["lat"] >= lat2)
        keep.append(df[mask])
    clipped = pd.concat(keep) if keep else pd.DataFrame(columns=df.columns)
    clipped.drop_duplicates(subset=["mmsi", "postime"], inplace=True)
    clipped.sort_values(["mmsi", "postime"], inplace=True)
    clipped.to_csv(dst, index=False, encoding="utf-8-sig")
    return len(df), len(clipped)


def collect_points(input_dir: Path) -> List[Dict]:
    points: List[Dict] = []
    for file in input_dir.rglob("*.jsonl"):
        with file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(json.loads(line))
                except Exception:
                    continue
    for file in input_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(file)
            points.extend(df.to_dict(orient="records"))
        except Exception:
            continue
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
    result.sort(key=lambda x: (str(x.get("mmsi")), str(x.get("postime"))))
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


def audit_simple(output_dir: Path, start: str, end: str, slice_hours: int):
    mmsi_list = dl.load_mmsi_csv(output_dir / "mmsi.csv") if (output_dir / "mmsi.csv").exists() else []
    slices = list(dl.split_time_range(dl.parse_time(start), dl.parse_time(end), slice_hours))
    rows = []
    for m in mmsi_list:
        missing = 0
        existing = 0
        for s, e in slices:
            token = f"{s.strftime('%Y%m%dT%H%M%S')}__{e.strftime('%Y%m%dT%H%M%S')}"
            jp = output_dir / "raw_tracks" / m / f"{token}.jsonl"
            cp = output_dir / "raw_tracks_csv" / m / f"{token}.csv"
            if jp.exists() or cp.exists():
                existing += 1
            else:
                missing += 1
        rows.append({"mmsi": m, "expected_slices": len(slices), "existing_slices": existing, "missing_slices": missing})
    cov_path = output_dir / "coverage_report.csv"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    with cov_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["mmsi", "expected_slices", "existing_slices", "missing_slices"])
        writer.writeheader()
        writer.writerows(rows)
    miss_total = sum(r["missing_slices"] for r in rows)
    miss_mmsi = sum(1 for r in rows if r["missing_slices"] > 0)
    summary = output_dir / "audit_summary.md"
    summary.write_text(
        f"# Audit Summary\n- mmsi: {len(mmsi_list)}\n- missing_slices total: {miss_total}\n- missing_mmsi: {miss_mmsi}\n",
        encoding="utf-8",
    )
    return miss_total, miss_mmsi


def compare_ls3(ls3_path: Path, ours_path: Path, census_mmsi: List[str], out_dir: Path, ts: str, selection_bbox: List[str]):
    ensure_dir(out_dir)
    if not ours_path.exists():
        return None, None
    ls = pd.read_csv(ls3_path)
    ours = pd.read_csv(ours_path)
    for df in (ls, ours):
        df["postime_dt"] = pd.to_datetime(df["postime"])
        df["postime_5min"] = df["postime_dt"].dt.floor("5min")
        df["key"] = df["mmsi"].astype(str) + "|" + df["postime_5min"].astype(str)
    ls_keys = set(ls["key"])
    our_keys = set(ours["key"])
    inter = ls_keys & our_keys
    coverage = len(inter) / len(ls_keys) if ls_keys else 0.0
    metrics_path = out_dir / f"ls3_compare_metrics_{ts}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ls3_keys", "our_keys", "intersection", "ls3_minus_our", "our_minus_ls3", "coverage"])
        writer.writerow([len(ls_keys), len(our_keys), len(inter), len(ls_keys - our_keys), len(our_keys - ls_keys), coverage])
    # Coverage by mmsi
    cov_rows = []
    ls_group = ls.groupby("mmsi")
    our_group = ours.groupby("mmsi")
    for mmsi, g in ls_group:
        ls_set = set(g["key"])
        our_set = set(our_group.get_group(mmsi)["key"]) if mmsi in our_group.groups else set()
        cov_rows.append({"mmsi": mmsi, "ls3_keys": len(ls_set), "our_keys": len(our_set), "intersection": len(ls_set & our_set)})
    cov_df = pd.DataFrame(cov_rows)
    cov_df["coverage"] = cov_df.apply(lambda r: (r["intersection"] / r["ls3_keys"]) if r["ls3_keys"] else 0.0, axis=1)
    top_cov = cov_df.sort_values("coverage", ascending=False).head(5)
    low_cov = cov_df.sort_values("coverage", ascending=True).head(5)
    # Missing by day
    ls_missing = ls_keys - our_keys
    missing_day = {}
    for k in ls_missing:
        _, ts_day = k.split("|")
        day = ts_day.split(" ")[0]
        missing_day[day] = missing_day.get(day, 0) + 1
    missing_day_sorted = sorted(missing_day.items(), key=lambda x: x[1], reverse=True)[:5]
    # MMSI sets
    census_set = set(census_mmsi)
    ls_mmsi_set = set(ls["mmsi"].astype(str))
    ls3_only = sorted(ls_mmsi_set - census_set)
    our_only = sorted(census_set - ls_mmsi_set)
    report_path = out_dir / f"ls3_compare_report_{ts}.md"
    lines = [
        f"# LS3 Compare ({ts})",
        f"- |LS3 keys|={len(ls_keys)} |OUR keys|={len(our_keys)}",
        f"- Intersection={len(inter)}",
        f"- Missing (LS3-OUR)={len(ls_keys - our_keys)}",
        f"- Extra (OUR-LS3)={len(our_keys - ls_keys)}",
        f"- Coverage={coverage:.2%}",
        "",
        "## Selection vs LS3 MMSI (census only)",
        f"- selection_bbox: {selection_bbox}",
        f"- census_mmsi_count: {len(census_set)}",
        f"- ls3_mmsi_count: {len(ls_mmsi_set)}",
        f"- ls3_only_mmsi: {ls3_only}",
        f"- our_only_mmsi: {our_only}",
        "",
        "## Coverage by MMSI (top/bottom)",
    ]
    lines.append("- top:")
    for _, row in top_cov.iterrows():
        lines.append(f"  - {row.mmsi}: {row.coverage:.2%} ({int(row.intersection)}/{int(row.ls3_keys)})")
    lines.append("- bottom:")
    for _, row in low_cov.iterrows():
        lines.append(f"  - {row.mmsi}: {row.coverage:.2%} ({int(row.intersection)}/{int(row.ls3_keys)})")
    lines.append("")
    lines.append("## Missing by day (LS3-OUR, top5)")
    for day, cnt in missing_day_sorted:
        lines.append(f"- {day}: {cnt}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics_path, report_path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline(args):
    start = dl.parse_time(args.start)
    end = dl.parse_time(args.end)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Census
    mmsi_path, census_path = census_mmsi(start, end, args.region, args.tile_deg, args.page_size, args.client_id, args.client_secret, out_dir)

    # Harvest
    dl_cfg = argparse.Namespace(
        start=args.start,
        end=args.end,
        region=args.region,
        output_dir=str(out_dir),
        page_size=args.page_size,
        slice_hours=args.slice_hours,
        slice_threshold=args.slice_threshold,
        client_id=args.client_id,
        client_secret=args.client_secret,
        output_format="csv",
        raw_dir="raw_tracks",
        raw_csv_dir="raw_tracks_csv",
        mmsi_csv=str(mmsi_path),
        interactive=False,
    )
    dl.run_downloader(dl_cfg)

    # Merge
    merged_path = out_dir / "AIS_merged.csv"
    all_points = collect_points(out_dir)
    deduped = dedupe_points(all_points)
    save_csv(deduped, merged_path)

    # Clip (skip when ls3_benchmark_mode)
    clipped_path = out_dir / "AIS_clipped_merged.csv"
    clip_in_rows = clip_out_rows = 0
    do_clip = args.clip_to_bbox and not args.ls3_benchmark_mode
    if do_clip:
        clip_in_rows, clip_out_rows = clip_csv(merged_path, args.region, clipped_path)

    # Audit
    audit_missing, audit_mmsi = audit_simple(out_dir, args.start, args.end, args.slice_hours)

    # Compare LS3
    compare_target = merged_path if args.ls3_benchmark_mode or not do_clip else clipped_path
    metrics_path, report_path = compare_ls3(Path(args.ls3_path), compare_target, dl.load_mmsi_csv(mmsi_path), out_dir / "logs" / "benchmark", ts, args.region)

    # Manifest
    manifest = {
        "start": args.start,
        "end": args.end,
        "region": args.region,
        "tile_deg": args.tile_deg,
        "slice_hours": args.slice_hours,
        "page_size": args.page_size,
        "slice_threshold": args.slice_threshold,
        "mmsi_registry": str(mmsi_path),
        "census_stats": str(census_path),
        "merged": str(merged_path),
        "clipped": str(clipped_path),
        "ls3_compare_metrics": str(metrics_path) if metrics_path else "",
        "ls3_compare_report": str(report_path) if report_path else "",
        "selection_bbox": args.region,
        "clip_enabled": False if args.ls3_benchmark_mode else args.clip_to_bbox,
        "audit_missing_slices": audit_missing,
        "audit_missing_mmsi": audit_mmsi,
        "clip_in_rows": clip_in_rows,
        "clip_out_rows": clip_out_rows,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def wizard():
    def ask(prompt, default=None):
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default
    start = ask("Start time", "2024-03-01 00:00:00")
    end = ask("End time", "2024-03-03 00:00:00")
    regions = []
    print("Enter bbox lon1,lat1,lon2,lat2 per line (blank to finish):")
    while True:
        v = input("> ").strip()
        if not v:
            break
        regions.append(v)
    if not regions:
        regions = ["50,75,55,70"]
    return argparse.Namespace(
        start=start,
        end=end,
        region=regions,
        tile_deg=1.0,
        slice_hours=6,
        page_size=100,
        slice_threshold=9000,
        output_dir="demo_pipeline_ls3",
        ls3_path="LS3.csv",
        clip_to_bbox=True,
        client_id=os.getenv("MYVESSEL_CLIENT_ID"),
        client_secret=os.getenv("MYVESSEL_CLIENT_SECRET"),
    )


def main():
    parser = argparse.ArgumentParser(description="One-click AIS pipeline: census -> harvest -> merge -> clip -> audit -> compare LS3")
    parser.add_argument("--start", default="2024-03-01 00:00:00")
    parser.add_argument("--end", default="2024-03-15 00:00:00")
    parser.add_argument("--region", action="append", default=["50,75,55,70"])
    parser.add_argument("--tile-deg", type=float, default=1.0)
    parser.add_argument("--slice-hours", type=int, default=6)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--slice-threshold", type=int, default=9000)
    parser.add_argument("--output-dir", default="demo_pipeline_ls3")
    parser.add_argument("--ls3-path", default="LS3.csv")
    parser.add_argument("--clip-to-bbox", action="store_true", default=True)
    parser.add_argument("--ls3-benchmark-mode", action="store_true", help="Use selection bbox only for census; disable clip; compare on merged.")
    parser.add_argument("--client-id", default=os.getenv("MYVESSEL_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("MYVESSEL_CLIENT_SECRET"))
    parser.add_argument("--interactive", "--wizard", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        args = wizard()
    if not args.client_id or not args.client_secret:
        print("Missing MYVESSEL_CLIENT_ID / MYVESSEL_CLIENT_SECRET")
        return
    start_time = time.time()
    run_pipeline(args)
    print(f"Elapsed: {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()
