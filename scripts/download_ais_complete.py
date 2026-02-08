import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


TOKEN_URL = "https://svc.data.myvessel.cn/ada/oauth/token"
REGION_EVENTS_URL = "https://svc.data.myvessel.cn/sdc-tob/v1/mkt/vessels/events/cross/region/his/page"
TRACK_URL = "https://svc.data.myvessel.cn/sdc-tob/v1/mkt/ais/track"

DEFAULT_PAGE_SIZE = 100
DEFAULT_SLICE_HOURS = 6
DEFAULT_SLICE_THRESHOLD = 9000
DEFAULT_OUTPUT_FORMAT = "both"
DEFAULT_RAW_DIR = "raw_tracks"
DEFAULT_RAW_CSV_DIR = "raw_tracks_csv"
DEFAULT_CENSUS_CHUNK_DAYS = 30


def isoformat(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def parse_time(text: str) -> dt.datetime:
    cleaned = text.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return dt.datetime.fromisoformat(cleaned)


def is_number(val) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def atomic_write(path: Path, data: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding, newline="") as f:
        f.write(data)
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def load_config(config_path: Path) -> Dict:
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config_path: Path, config: Dict) -> None:
    safe = {k: v for k, v in config.items() if k != "client_secret"}
    atomic_write(config_path, json.dumps(safe, ensure_ascii=False, indent=2))


class ApiClient:
    def __init__(self, client_id: str, client_secret: str):
        if not client_id or not client_secret:
            raise ValueError("client_id/client_secret are required (env MYVESSEL_CLIENT_ID / MYVESSEL_CLIENT_SECRET or config).")
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()
        self.token = None
        self.refresh_token()

    def refresh_token(self):
        resp = self.session.post(
            TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        self.token = body["access_token"]

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        headers = kwargs.pop("headers", {}) or {}
        all_headers = {"User-Agent": "ais-downloader/complete"}
        if self.token:
            all_headers["Authorization"] = f"Bearer {self.token}"
        all_headers.update(headers)
        kwargs["headers"] = all_headers

        resp = self._request_with_retry(method, url, **kwargs)
        if resp.status_code in (401, 403):
            self.refresh_token()
            all_headers["Authorization"] = f"Bearer {self.token}"
            kwargs["headers"] = all_headers
            resp = self._request_with_retry(method, url, **kwargs)
        self._ensure_success(resp)
        return resp

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        max_retries: int = 3,
        min_sleep: float = 0.2,
        max_sleep: float = 0.5,
        backoff_base: float = 0.8,
        allow_statuses: Tuple[int, ...] = (401, 403),
        **kwargs,
    ) -> requests.Response:
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in allow_statuses:
                    return resp
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"server/status {resp.status_code}", response=resp)
                resp.raise_for_status()
                time.sleep(random.uniform(min_sleep, max_sleep))
                return resp
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                if attempt == max_retries:
                    break
                retry_after = 0
                if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None):
                    ra = exc.response.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        retry_after = int(ra)
                delay = retry_after if retry_after else backoff_base * (2 ** (attempt - 1))
                time.sleep(delay + random.uniform(min_sleep, max_sleep))
        if isinstance(last_exc, requests.HTTPError) and getattr(last_exc, "response", None):
            raise last_exc
        raise last_exc

    def _ensure_success(self, resp: requests.Response) -> None:
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            return
        success = data.get("success", True)
        code = data.get("code")
        if not success or (code not in (None, "", "0")):
            msg = data.get("message") or f"API error code={code}"
            raise RuntimeError(msg)


def extract_list_from_data(data) -> List[Dict]:
    if isinstance(data, dict):
        if isinstance(data.get("content"), list):
            return data["content"]
        if isinstance(data.get("records"), list):
            return data["records"]
    if isinstance(data, list):
        return data
    return []


def fetch_region_events(
    api: ApiClient,
    region: Dict,
    start: dt.datetime,
    end: dt.datetime,
    page_size: int = 100,
    progress_prefix: str = "",
) -> Tuple[List[Dict], int]:
    page_num = 1
    all_rows: List[Dict] = []
    total_recorded = 0
    while True:
        payload = {
            "dwt": 0,
            "teu": 0,
            "grt": 0,
            "vesselSubType": [],
            "teu2": 40000,
            "grt2": 40000,
            "startTime": isoformat(start),
            "endTime": isoformat(end),
            "page": {"pageSize": page_size, "pageNum": page_num},
            "region": region,
            "dwt2": 400000,
        }
        resp = api.request("post", REGION_EVENTS_URL, json=payload, timeout=90)
        body = resp.json()
        data = body.get("data") or {}
        content = extract_list_from_data(data)
        all_rows.extend(content)
        total_recorded += len(content)
        if page_num == 1 or page_num % 10 == 0 or len(content) < page_size:
            prefix = f"{progress_prefix} " if progress_prefix else ""
            print(
                f"{prefix}census page={page_num} fetched={len(content)} total={total_recorded}",
                flush=True,
            )

        total_pages = None
        if isinstance(data, dict):
            total_pages = data.get("totalPages") or data.get("pages")
        if not content or len(content) < page_size or (total_pages and page_num >= total_pages):
            break
        page_num += 1
    return all_rows, total_recorded


def extract_mmsi(rows: Iterable[Dict]) -> List[str]:
    mmsis = []
    seen = set()
    for row in rows:
        val = row.get("mmsi")
        if val is None:
            continue
        text = str(val).strip()
        if text.isdigit() and text not in seen:
            seen.add(text)
            mmsis.append(text)
    return mmsis


def save_mmsi_csv(path: Path, mmsis: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mmsi"])
        for mmsi in mmsis:
            writer.writerow([mmsi])
    tmp.replace(path)


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


def split_time_range(start: dt.datetime, end: dt.datetime, slice_hours: int) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    delta = dt.timedelta(hours=slice_hours)
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + delta)
        yield cursor, nxt
        cursor = nxt


def split_region_query_range(
    start: dt.datetime,
    end: dt.datetime,
    max_days: int = DEFAULT_CENSUS_CHUNK_DAYS,
) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    """Split census query into shorter windows to avoid API 3-month limit errors."""
    delta = dt.timedelta(days=max(1, int(max_days)))
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + delta)
        yield cursor, nxt
        cursor = nxt


def parse_track_points(body: Dict) -> List[Dict]:
    data = body.get("data")
    points = extract_list_from_data(data)
    return points or []


def fetch_track_points(
    api: ApiClient,
    mmsi: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
    *,
    slice_threshold: int = 9000,
) -> List[Dict]:
    collected: List[Dict] = []
    cursor = window_start
    while cursor < window_end:
        payload = {
            "mmsi": mmsi,
            "startTime": isoformat(cursor),
            "endTime": isoformat(window_end),
        }
        resp = api.request("post", TRACK_URL, json=payload, timeout=20)
        points = parse_track_points(resp.json())
        cleaned = []
        for p in points:
            lon = p.get("lon")
            lat = p.get("lat")
            if not (is_number(lon) and is_number(lat)):
                continue
            cleaned.append(p)
        collected.extend(cleaned)
        if not points:
            break
        if len(points) >= slice_threshold:
            last_time = points[-1].get("postime")
            if not last_time:
                break
            cursor = parse_time(str(last_time)) + dt.timedelta(seconds=1)
            if cursor >= window_end:
                break
            continue
        break
    deduped = []
    seen = set()
    for p in collected:
        key = (str(p.get("mmsi")), str(p.get("postime")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def region_from_bbox(bbox: str) -> Dict:
    lon1, lat1, lon2, lat2 = [float(x) for x in bbox.split(",")]
    return {
        "regionType": "rectangle",
        "leftTop": {"lon": lon1, "lat": lat1},
        "rightBottom": {"lon": lon2, "lat": lat2},
    }


def _normalize_bbox(bbox: str) -> Tuple[float, float, float, float]:
    lon1, lat1, lon2, lat2 = [float(x) for x in bbox.split(",")]
    left = min(lon1, lon2)
    right = max(lon1, lon2)
    top = max(lat1, lat2)
    bottom = min(lat1, lat2)
    return left, top, right, bottom


def tile_bbox(bbox: str, tile_deg: float = 0.1) -> List[str]:
    left, top, right, bottom = _normalize_bbox(bbox)
    step = max(float(tile_deg), 0.02)
    tiles: List[str] = []
    cur_lat_top = top
    while cur_lat_top > bottom:
        cur_lat_bottom = max(bottom, cur_lat_top - step)
        cur_lon_left = left
        while cur_lon_left < right:
            cur_lon_right = min(right, cur_lon_left + step)
            tiles.append(
                f"{cur_lon_left:.6f},{cur_lat_top:.6f},{cur_lon_right:.6f},{cur_lat_bottom:.6f}"
            )
            cur_lon_left = cur_lon_right
        cur_lat_top = cur_lat_bottom
    return tiles


def save_points_csv(path: Path, points: List[Dict]) -> None:
    if not points:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, "", encoding="utf-8-sig")
        return
    cols = ["mmsi", "postime", "lon", "lat", "status", "eta", "dest", "draught", "cog", "hdg", "sog", "rot"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in points:
            writer.writerow({k: row.get(k) for k in cols})
    tmp.replace(path)


def run_downloader(args):
    output_dir = Path(args.output_dir)
    config_path = output_dir / "config.json"
    config_defaults = load_config(config_path)

    client_id = args.client_id or os.getenv("MYVESSEL_CLIENT_ID") or config_defaults.get("client_id")
    client_secret = args.client_secret or os.getenv("MYVESSEL_CLIENT_SECRET") or config_defaults.get("client_secret")
    api = ApiClient(client_id, client_secret)

    raw_dir = output_dir / (args.raw_dir or DEFAULT_RAW_DIR)
    raw_csv_dir = output_dir / (args.raw_csv_dir or DEFAULT_RAW_CSV_DIR)

    start_dt = parse_time(args.start)
    end_dt = parse_time(args.end)

    # Region events -> MMSI
    all_region_rows: List[Dict] = []
    for bbox in args.region:
        tiles = tile_bbox(bbox, args.census_tile_deg)
        print(f"Region {bbox}: tile_count={len(tiles)} tile_deg={args.census_tile_deg}", flush=True)
        region_total = 0
        for tile in tiles:
            region = region_from_bbox(tile)
            tile_total = 0
            for qs, qe in split_region_query_range(start_dt, end_dt, DEFAULT_CENSUS_CHUNK_DAYS):
                print(
                    f"Tile {tile}: query-start {isoformat(qs)} -> {isoformat(qe)}",
                    flush=True,
                )
                rows, total = fetch_region_events(
                    api,
                    region,
                    qs,
                    qe,
                    page_size=args.page_size,
                    progress_prefix=f"[{tile} {isoformat(qs)}~{isoformat(qe)}]",
                )
                all_region_rows.extend(rows)
                tile_total += total
                region_total += total
                print(
                    f"Tile {tile}: query {isoformat(qs)} -> {isoformat(qe)} "
                    f"records {total}, pagesize {args.page_size}",
                    flush=True,
                )
            print(f"Tile {tile}: total records {tile_total}", flush=True)
        print(f"Region {bbox}: total records {region_total}, pagesize {args.page_size}", flush=True)
    mmsis = extract_mmsi(all_region_rows)
    mmsi_path = output_dir / "mmsi.csv"
    save_mmsi_csv(mmsi_path, mmsis)
    print(f"MMSI saved: {len(mmsis)} -> {mmsi_path}")

    target_mmsis = load_mmsi_csv(mmsi_path)
    failed_windows: List[Tuple[str, str, str]] = []
    summary_counts = 0

    needs_jsonl = args.output_format in ("jsonl", "both")
    needs_csv = args.output_format in ("csv", "both")

    for mmsi in target_mmsis:
        for slice_start, slice_end in split_time_range(start_dt, end_dt, args.slice_hours):
            time_token = f"{slice_start.strftime('%Y%m%dT%H%M%S')}__{slice_end.strftime('%Y%m%dT%H%M%S')}"
            jsonl_path = raw_dir / mmsi / f"{time_token}.jsonl"
            csv_path = raw_csv_dir / mmsi / f"{time_token}.csv"
            has_jsonl = jsonl_path.exists() if needs_jsonl else True
            has_csv = csv_path.exists() if needs_csv else True
            if has_jsonl and has_csv:
                print(f"Skip existing {mmsi} {time_token}")
                continue
            try:
                points = fetch_track_points(api, mmsi, slice_start, slice_end, slice_threshold=args.slice_threshold)
                if needs_jsonl:
                    atomic_write_jsonl(jsonl_path, points)
                if needs_csv:
                    save_points_csv(csv_path, points)
                summary_counts += len(points)
                print(f"MMSI {mmsi} slice {slice_start} -> {slice_end}: {len(points)} points")
            except Exception as exc:  # pragma: no cover
                failed_windows.append((mmsi, isoformat(slice_start), isoformat(slice_end)))
                print(f"Failed {mmsi} {slice_start} -> {slice_end}: {exc}")

    if failed_windows:
        fail_path = output_dir / "failed_windows.csv"
        with fail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["mmsi", "slice_start", "slice_end"])
            writer.writerows(failed_windows)
        print(f"Failed windows saved to {fail_path}")
    print(f"Total track points downloaded: {summary_counts}")

    save_config(
        config_path,
        {
            "start": args.start,
            "end": args.end,
            "region": args.region,
            "output_dir": args.output_dir,
            "page_size": args.page_size,
            "slice_hours": args.slice_hours,
            "slice_threshold": args.slice_threshold,
            "output_format": args.output_format,
            "raw_dir": args.raw_dir,
            "raw_csv_dir": args.raw_csv_dir,
            "census_tile_deg": args.census_tile_deg,
            "client_id": client_id,
        },
    )


def prompt_with_default(prompt: str, default: Optional[str]) -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or (default or "")


def validate_bbox(text: str) -> str:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("bbox must have 4 comma-separated numbers")
    [float(p) for p in parts]
    return text


def wizard(args):
    output_dir = Path(args.output_dir or ".")
    config_path = output_dir / "config.json"
    prev = load_config(config_path)

    start = prompt_with_default("Start time (YYYY-MM-DD HH:MM:SS)", prev.get("start"))
    end = prompt_with_default("End time (YYYY-MM-DD HH:MM:SS)", prev.get("end"))

    regions: List[str] = []
    default_regions = prev.get("region") or []
    print("Enter region bbox lon1,lat1,lon2,lat2 (blank to finish).")
    if default_regions:
        print(f"Default regions: {default_regions}")
    while True:
        val = input("> ").strip()
        if not val:
            if not regions and default_regions:
                regions.extend(default_regions)
            break
        try:
            regions.append(validate_bbox(val))
        except Exception as exc:
            print(f"Invalid bbox: {exc}")
            continue

    page_size = prompt_with_default("Page size", str(prev.get("page_size", DEFAULT_PAGE_SIZE)))
    slice_hours = prompt_with_default("Slice hours", str(prev.get("slice_hours", DEFAULT_SLICE_HOURS)))
    slice_threshold = prompt_with_default("Slice threshold", str(prev.get("slice_threshold", DEFAULT_SLICE_THRESHOLD)))

    output_dir_str = prompt_with_default("Output dir", str(prev.get("output_dir", output_dir)))
    raw_dir = prompt_with_default("Raw jsonl dir name", prev.get("raw_dir", DEFAULT_RAW_DIR))
    raw_csv_dir = prompt_with_default("Raw csv dir name", prev.get("raw_csv_dir", DEFAULT_RAW_CSV_DIR))
    output_format = prompt_with_default("Output format (jsonl/csv/both)", prev.get("output_format", DEFAULT_OUTPUT_FORMAT))

    client_id = prompt_with_default("Client ID", os.getenv("MYVESSEL_CLIENT_ID") or prev.get("client_id") or "")
    client_secret = prompt_with_default("Client Secret", os.getenv("MYVESSEL_CLIENT_SECRET") or prev.get("client_secret") or "")

    cfg = {
        "start": start,
        "end": end,
        "region": regions,
        "output_dir": output_dir_str,
        "page_size": int(page_size),
        "slice_hours": int(slice_hours),
        "slice_threshold": int(slice_threshold),
        "output_format": output_format or DEFAULT_OUTPUT_FORMAT,
        "raw_dir": raw_dir,
        "raw_csv_dir": raw_csv_dir,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    print("\nResolved configuration:")
    for k, v in cfg.items():
        if k == "client_secret":
            print(f"  {k}: {'***' if v else ''}")
        else:
            print(f"  {k}: {v}")
    confirm = input("Proceed? (Y/n): ").strip().lower()
    if confirm and confirm not in ("y", "yes"):
        print("Aborted by user.")
        sys.exit(1)

    args.start = cfg["start"]
    args.end = cfg["end"]
    args.region = cfg["region"]
    args.output_dir = cfg["output_dir"]
    args.page_size = cfg["page_size"]
    args.slice_hours = cfg["slice_hours"]
    args.slice_threshold = cfg["slice_threshold"]
    args.output_format = cfg["output_format"]
    args.raw_dir = cfg["raw_dir"]
    args.raw_csv_dir = cfg["raw_csv_dir"]
    args.client_id = cfg["client_id"]
    args.client_secret = cfg["client_secret"]
    run_downloader(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete AIS downloader with pagination, slicing, resume, and wizard mode.")
    parser.add_argument("--start", help="Start time, e.g. 2024-06-01 00:00:00")
    parser.add_argument("--end", help="End time, e.g. 2024-06-02 00:00:00")
    parser.add_argument(
        "--region",
        action="append",
        help="Bounding box as lon1,lat1,lon2,lat2 (leftTop lon/lat, rightBottom lon/lat). Can repeat.",
    )
    parser.add_argument("--output-dir", help="Directory to store outputs and raw slices.", default="output")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Page size for region events pagination.")
    parser.add_argument("--slice-hours", type=int, default=DEFAULT_SLICE_HOURS, help="Hours per time slice for track download.")
    parser.add_argument("--slice-threshold", type=int, default=DEFAULT_SLICE_THRESHOLD, help="Threshold to continue slicing inside a window.")
    parser.add_argument("--client-id", help="API client_id (or set env MYVESSEL_CLIENT_ID).")
    parser.add_argument("--client-secret", help="API client_secret (or set env MYVESSEL_CLIENT_SECRET).")
    parser.add_argument("--output-format", choices=["jsonl", "csv", "both"], default=DEFAULT_OUTPUT_FORMAT, help="Slice output format.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Subdirectory for jsonl slices.")
    parser.add_argument("--raw-csv-dir", default=DEFAULT_RAW_CSV_DIR, help="Subdirectory for csv slices.")
    parser.add_argument("--census-tile-deg", type=float, default=0.1, help="Spatial tile size (degrees) for region census requests.")
    parser.add_argument("--interactive", "--wizard", action="store_true", help="Start interactive wizard.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.interactive:
        wizard(args)
    else:
        if not args.start or not args.end or not args.region:
            print("Non-interactive mode requires --start --end and at least one --region.")
            sys.exit(1)
        run_downloader(args)


if __name__ == "__main__":
    main()
