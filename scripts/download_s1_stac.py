from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


ROOT = Path(__file__).resolve().parents[1]
STAC_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def _request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    retry_wait: float = 2.0,
    timeout: int = 60,
    **kwargs: Any,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for i in range(1, max_retries + 1):
        try:
            resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
            # Retry 5xx and selected transient 429.
            if resp.status_code >= 500 or resp.status_code == 429:
                if i < max_retries:
                    wait = retry_wait * i
                    print(f"[retry] {method} {url} status={resp.status_code} attempt={i}/{max_retries} wait={wait:.1f}s")
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if i >= max_retries:
                break
            wait = retry_wait * i
            print(f"[retry] {method} {url} error={exc} attempt={i}/{max_retries} wait={wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"request failed after retries: {method} {url} -> {last_exc}")


def _pick_asset(feature: Dict[str, Any], asset_kind: str) -> Optional[str]:
    assets = feature.get("assets") or {}
    if asset_kind == "product":
        product = assets.get("Product")
        if isinstance(product, dict) and isinstance(product.get("href"), str):
            return product["href"]
        return None

    for key in ("quicklook", "thumbnail", "overview"):
        node = assets.get(key)
        if isinstance(node, dict) and isinstance(node.get("href"), str):
            return node["href"]
    for v in assets.values():
        if isinstance(v, dict) and isinstance(v.get("href"), str) and "http" in v["href"]:
            return v["href"]
    return None


def _asset_ext(asset_kind: str, href: str) -> str:
    if asset_kind == "product":
        return ".zip"
    lower = href.lower()
    if ".png" in lower:
        return ".png"
    if ".tif" in lower or ".tiff" in lower:
        return ".tif"
    return ".jpg"


def _search_s1(
    bbox: str,
    start: str,
    end: str,
    max_items: int,
    product_type: str = "GRD",
) -> List[Dict[str, Any]]:
    lon1, lat1, lon2, lat2 = [float(x) for x in bbox.split(",")]
    body = {
        "collections": ["sentinel-1-grd"],
        "bbox": [lon1, lat1, lon2, lat2],
        "datetime": f"{start}/{end}",
        "limit": max_items,
        "query": {"product:type": {"eq": product_type}},
    }
    resp = _request_with_retry("POST", STAC_SEARCH_URL, json=body, timeout=60)
    data = resp.json()
    features = data.get("features") or []
    if not features:
        # Fallback: some STAC deployments do not index product:type consistently.
        body_no_query = {
            "collections": ["sentinel-1-grd"],
            "bbox": [lon1, lat1, lon2, lat2],
            "datetime": f"{start}/{end}",
            "limit": max_items,
        }
        resp2 = _request_with_retry("POST", STAC_SEARCH_URL, json=body_no_query, timeout=60)
        data2 = resp2.json()
        features = data2.get("features") or []
    features.sort(key=lambda x: (x.get("properties") or {}).get("datetime", ""), reverse=False)
    return features


def _get_cdse_token(username: str, password: str) -> str:
    body = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    resp = _request_with_retry("POST", CDSE_TOKEN_URL, data=body, timeout=60)
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("failed to get Copernicus access token")
    return token


def _download(
    url: str,
    out: Path,
    headers: Optional[Dict[str, str]] = None,
    resume: bool = True,
) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = out.stat().st_size if out.exists() else 0
    req_headers: Dict[str, str] = dict(headers or {})
    mode = "wb"
    if resume and existing > 0:
        req_headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with _request_with_retry(
        "GET",
        url,
        headers=req_headers,
        stream=True,
        timeout=300,
        max_retries=6,
        retry_wait=3.0,
    ) as r:
        if r.status_code == 416 and out.exists() and out.stat().st_size > 0:
            return False
        if r.status_code == 200 and mode == "ab":
            # Range not supported: restart to avoid duplicate bytes.
            existing = 0
            mode = "wb"
        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total_bytes = (int(total) + existing) if total and total.isdigit() else None
        downloaded = existing
        last_print = 0.0
        started = time.time()

        with out.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 1.0:
                    if total_bytes:
                        pct = (downloaded / total_bytes) * 100.0
                        print(f"[download] {out.name} {downloaded}/{total_bytes} bytes ({pct:.1f}%)")
                    else:
                        print(f"[download] {out.name} {downloaded} bytes")
                    last_print = now
        elapsed = max(time.time() - started, 0.001)
        speed_mb_s = (downloaded - existing) / 1024 / 1024 / elapsed
        if total_bytes:
            print(f"[download] done {out.name} {downloaded}/{total_bytes} bytes @ {speed_mb_s:.2f} MB/s")
        else:
            print(f"[download] done {out.name} {downloaded} bytes @ {speed_mb_s:.2f} MB/s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Search and download Sentinel-1 assets from Copernicus STAC")
    parser.add_argument("--bbox", default="121.90,30.50,122.25,30.75", help="lon1,lat1,lon2,lat2")
    parser.add_argument("--start", default="2024-04-01T00:00:00Z")
    parser.add_argument("--end", default="2024-06-30T23:59:59Z")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--product-type", default="GRD")
    parser.add_argument("--asset", choices=["quicklook", "product"], default="quicklook")
    parser.add_argument("--manifest", default="outputs/logs/s1_stac_items.json")
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copernicus-user", default=os.getenv("COPERNICUS_USER", ""))
    parser.add_argument("--copernicus-password", default=os.getenv("COPERNICUS_PASSWORD", ""))
    args = parser.parse_args()

    if not args.download_dir:
        if args.asset == "product":
            args.download_dir = "data/raw/s1/grd_zip"
        else:
            args.download_dir = "data/raw/s1/quicklook"

    features = _search_s1(
        bbox=args.bbox,
        start=args.start,
        end=args.end,
        max_items=max(1, min(args.max_items, 1000)),
        product_type=args.product_type,
    )
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    auth_headers: Dict[str, str] = {}
    if args.download and args.asset == "product":
        if args.copernicus_user and args.copernicus_password:
            token = _get_cdse_token(args.copernicus_user, args.copernicus_password)
            auth_headers = {"Authorization": f"Bearer {token}"}
            print("[init] Copernicus token OK")
        else:
            print("[warn] product download requested but COPERNICUS_USER/PASSWORD not set; will skip downloads")

    rows: List[Dict[str, Any]] = []
    dl_count = 0
    skipped_no_auth = 0
    for idx, f in enumerate(features, 1):
        props = f.get("properties") or {}
        fid = f.get("id", "")
        dt = props.get("datetime", "")
        asset = _pick_asset(f, args.asset)
        local_file = None
        downloaded = False
        err = None

        if asset and args.download:
            needs_auth = args.asset == "product"
            if needs_auth and not auth_headers:
                skipped_no_auth += 1
            else:
                ext = _asset_ext(args.asset, asset)
                out = ROOT / args.download_dir / f"{fid}{ext}"
                local_file = str(out)
                try:
                    print(f"[{idx}/{len(features)}] {fid}")
                    downloaded = _download(asset, out, headers=auth_headers, resume=args.resume)
                    if downloaded:
                        dl_count += 1
                except Exception as ex:
                    err = str(ex)
                    print(f"[error] {fid}: {err}")

        rows.append(
            {
                "id": fid,
                "datetime": dt,
                "asset_kind": args.asset,
                "sat:orbit_state": props.get("sat:orbit_state"),
                "sar:instrument_mode": props.get("sar:instrument_mode"),
                "sar:product_type": props.get("sar:product_type"),
                "asset_href": asset,
                "local_file": local_file,
                "downloaded_now": downloaded,
                "error": err,
            }
        )

    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] items={len(rows)} manifest={manifest_path}")
    if args.download:
        print(f"[done] downloaded_new={dl_count} skipped_no_auth={skipped_no_auth} dir={ROOT / args.download_dir}")


if __name__ == "__main__":
    main()
