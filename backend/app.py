from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
EVIDENCE_DIR = ROOT / "outputs" / "evidence_cards"
LOG_DIR = ROOT / "outputs" / "logs"
RAW_AIS_DIR = ROOT / "data" / "raw" / "ais"
PROCESSED_AIS_DIR = ROOT / "data" / "processed" / "ais_cleaned"
INTERIM_AIS_DIR = ROOT / "data" / "interim" / "ais_clean"
FRONTEND_DIST_DIR = ROOT / "frontend" / "dist"

DEFAULT_BBOX = "121.90,30.75,122.25,30.50"
DEFAULT_START = "2024-04-01 00:00:00"
DEFAULT_END = "2024-06-30 23:59:59"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _run_script(args: List[str]) -> None:
    cmd = [sys.executable] + args
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"script failed: {' '.join(args)}"
        raise RuntimeError(msg)


def _path_writable(path: Path) -> bool:
    try:
        return os.access(path, os.W_OK)
    except Exception:
        return False


def _tail(path: Path, max_lines: int = 40) -> List[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-max_lines:]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_records(df: pd.DataFrame, time_col: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    out = df.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.dropna(subset=[time_col])
        out[time_col] = out[time_col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def _count_files(folder: Path, pattern: str = "*", recursive: bool = False) -> int:
    if not folder.exists():
        return 0
    try:
        if recursive:
            return sum(1 for p in folder.rglob(pattern) if p.is_file())
        return sum(1 for p in folder.glob(pattern) if p.is_file())
    except Exception:
        return 0


def _bool_env(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    return len(value) > 0


def _log_activity(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "active_recently": False, "mtime": None}
    mtime = _mtime_iso(path)
    age_sec = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)
    return {
        "exists": True,
        "active_recently": bool(age_sec <= 300),
        "mtime": mtime,
    }


def _health_checks() -> Dict[str, Any]:
    metrics_files = {
        "congestion_curve": METRICS_DIR / "congestion_curve.csv",
        "waiting_time_by_day": METRICS_DIR / "waiting_time_by_day.csv",
        "waiting_time_summary": METRICS_DIR / "waiting_time_summary.csv",
        "congestion_forecast": METRICS_DIR / "congestion_forecast.csv",
        "waiting_forecast": METRICS_DIR / "waiting_forecast.csv",
        "vision_forecast": METRICS_DIR / "vision_forecast.csv",
        "yolo_observed": METRICS_DIR / "yolo_observed.csv",
    }
    metrics_state = {
        name: {"exists": p.exists(), "mtime": _mtime_iso(p), "size": (p.stat().st_size if p.exists() else 0)}
        for name, p in metrics_files.items()
    }

    ais_scan = _scan_latest_ais_file()
    s1_quicklook_count = _count_files(ROOT / "data" / "raw" / "s1" / "quicklook")
    s1_grd_zip_count = _count_files(ROOT / "data" / "raw" / "s1" / "grd_zip", "*.zip")
    s1_grd_png_count = _count_files(ROOT / "data" / "interim" / "s1_grd_png", "*.png")
    yolo_model = ROOT / "assets" / "models" / "sar_ship_yolov8n.pt"
    fallback_yolo_model = ROOT / "yolov8n.pt"

    dir_checks = {
        "outputs_metrics": {
            "path": str(METRICS_DIR),
            "exists": METRICS_DIR.exists(),
            "writable": _path_writable(METRICS_DIR) if METRICS_DIR.exists() else False,
        },
        "outputs_logs": {
            "path": str(LOG_DIR),
            "exists": LOG_DIR.exists(),
            "writable": _path_writable(LOG_DIR) if LOG_DIR.exists() else False,
        },
        "raw_ais": {
            "path": str(RAW_AIS_DIR),
            "exists": RAW_AIS_DIR.exists(),
            "writable": _path_writable(RAW_AIS_DIR) if RAW_AIS_DIR.exists() else False,
        },
        "raw_s1": {
            "path": str(ROOT / "data" / "raw" / "s1"),
            "exists": (ROOT / "data" / "raw" / "s1").exists(),
            "writable": _path_writable(ROOT / "data" / "raw" / "s1") if (ROOT / "data" / "raw" / "s1").exists() else False,
        },
    }

    activity = {
        "ais_download_log": _log_activity(LOG_DIR / "download_ais.log"),
        "s1_download_log": _log_activity(LOG_DIR / "download_s1_product.log"),
        "pipeline_log": _log_activity(LOG_DIR / "pipeline.log"),
    }

    with _jobs_lock:
        internal_jobs = {name: row.model_dump() for name, row in _jobs.items()}

    checks = {
        "runtime": {"python_version": sys.version.split()[0]},
        "credentials": {
            "myvessel_client_id_set": _bool_env("MYVESSEL_CLIENT_ID"),
            "myvessel_client_secret_set": _bool_env("MYVESSEL_CLIENT_SECRET"),
            "copernicus_user_set": _bool_env("COPERNICUS_USER"),
            "copernicus_password_set": _bool_env("COPERNICUS_PASSWORD"),
        },
        "filesystem": dir_checks,
        "data": {
            "ais_raw_file_count": int(ais_scan.get("count") or 0),
            "ais_latest_file": ais_scan.get("latest"),
            "s1_quicklook_count": s1_quicklook_count,
            "s1_grd_zip_count": s1_grd_zip_count,
            "s1_grd_png_count": s1_grd_png_count,
            "metrics": metrics_state,
        },
        "models": {
            "sar_yolo_model": {"path": str(yolo_model), "exists": yolo_model.exists()},
            "fallback_yolo_model": {"path": str(fallback_yolo_model), "exists": fallback_yolo_model.exists()},
        },
        "activity": activity,
        "jobs": internal_jobs,
        "pipeline_ready": {
            "has_ais_any": int(ais_scan.get("count") or 0) > 0,
            "has_s1_any": (s1_grd_zip_count > 0) or (s1_quicklook_count > 0),
            "has_cleaned_ais": _count_files(PROCESSED_AIS_DIR, "*.csv") > 0,
            "has_metrics_base": (METRICS_DIR / "congestion_curve.csv").exists(),
            "has_yolo_input": s1_grd_png_count > 0 or s1_quicklook_count > 0,
            "has_yolo_model": yolo_model.exists() or fallback_yolo_model.exists(),
        },
    }

    warnings: List[str] = []
    if not checks["credentials"]["myvessel_client_id_set"] or not checks["credentials"]["myvessel_client_secret_set"]:
        warnings.append("MyVessel credentials are not fully configured.")
    if not checks["credentials"]["copernicus_user_set"] or not checks["credentials"]["copernicus_password_set"]:
        warnings.append("Copernicus credentials are not fully configured.")
    if not checks["filesystem"]["outputs_metrics"]["exists"]:
        warnings.append("outputs/metrics directory is missing.")
    if not checks["pipeline_ready"]["has_yolo_model"]:
        warnings.append("YOLO model weights are missing.")

    ok = len(warnings) == 0
    status = "ok" if ok else "degraded"
    return {"ok": ok, "status": status, "warnings": warnings, "checks": checks}


class StartDownloadRequest(BaseModel):
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    bbox: str = DEFAULT_BBOX
    slice_hours: int = Field(default=6, ge=1, le=24)
    page_size: int = Field(default=100, ge=10, le=500)
    slice_threshold: int = Field(default=9000, ge=100)


class StartPipelineRequest(BaseModel):
    horizon_days: int = Field(default=24, ge=1, le=24)


class JobState(BaseModel):
    name: str
    running: bool
    pid: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    command: Optional[List[str]] = None
    log_file: Optional[str] = None


_jobs: Dict[str, JobState] = {}
_jobs_lock = threading.Lock()


def _watch_job(name: str, proc: subprocess.Popen[str]) -> None:
    rc = proc.wait()
    with _jobs_lock:
        old = _jobs.get(name)
        if not old:
            return
        _jobs[name] = JobState(
            **{
                **old.model_dump(),
                "running": False,
                "return_code": rc,
                "finished_at": _now_iso(),
            }
        )


def _start_job(name: str, command: List[str], log_file: Path) -> JobState:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _jobs_lock:
        cur = _jobs.get(name)
        if cur and cur.running:
            return cur
        log_fh = open(log_file, "a", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        state = JobState(
            name=name,
            running=True,
            pid=proc.pid,
            started_at=_now_iso(),
            finished_at=None,
            return_code=None,
            command=command,
            log_file=str(log_file),
        )
        _jobs[name] = state
        watcher = threading.Thread(target=_watch_job, args=(name, proc), daemon=True)
        watcher.start()
        return state


def _scan_latest_ais_file() -> Dict[str, Any]:
    dirs = [RAW_AIS_DIR / "raw_tracks_csv", RAW_AIS_DIR / "raw_tracks"]
    files: List[Path] = []
    for folder in dirs:
        if folder.exists():
            files.extend([p for p in folder.rglob("*") if p.is_file()])
    if not files:
        return {"count": 0, "latest": None, "latest_mtime": None}
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = files[0]
    return {
        "count": len(files),
        "latest": str(latest.relative_to(ROOT)),
        "latest_mtime": _mtime_iso(latest),
    }


def _find_ais_source_csv() -> Optional[Path]:
    candidates = [
        *sorted(PROCESSED_AIS_DIR.glob("*.csv")),
        *sorted(INTERIM_AIS_DIR.glob("*.csv")),
        *sorted((RAW_AIS_DIR).glob("*.csv")),
    ]
    return candidates[0] if candidates else None


def _frontend_index() -> Path:
    return FRONTEND_DIST_DIR / "index.html"


def _frontend_file_or_none(rel_path: str) -> Optional[Path]:
    if not rel_path:
        return None
    # Block path traversal and only allow files under dist/.
    target = (FRONTEND_DIST_DIR / rel_path).resolve()
    dist_root = FRONTEND_DIST_DIR.resolve()
    if dist_root not in target.parents and target != dist_root:
        return None
    if target.exists() and target.is_file():
        return target
    return None


app = FastAPI(title="MoorCaster API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Response:
    index = _frontend_index()
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/favicon.ico")
def favicon() -> Response:
    ico = _frontend_file_or_none("favicon.ico")
    if ico:
        return FileResponse(ico)
    return Response(status_code=204)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    payload = _health_checks()
    payload["time"] = _now_iso()
    return payload


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    tracked = {
        "congestion_curve": METRICS_DIR / "congestion_curve.csv",
        "waiting_time_by_day": METRICS_DIR / "waiting_time_by_day.csv",
        "waiting_time_summary": METRICS_DIR / "waiting_time_summary.csv",
        "congestion_forecast": METRICS_DIR / "congestion_forecast.csv",
        "waiting_forecast": METRICS_DIR / "waiting_forecast.csv",
        "vision_forecast": METRICS_DIR / "vision_forecast.csv",
    }
    file_state = {k: {"path": str(v), "mtime": _mtime_iso(v), "exists": v.exists()} for k, v in tracked.items()}
    mtimes = [v["mtime"] for v in file_state.values() if v["mtime"]]
    latest = max(mtimes) if mtimes else None
    return {
        "server_time": _now_iso(),
        "last_updated_utc": latest,
        "files": file_state,
        "ais_download": _scan_latest_ais_file(),
    }


@app.get("/api/series/congestion")
def series_congestion(granularity: str = "day") -> Dict[str, Any]:
    src = METRICS_DIR / "congestion_curve.csv"
    df = _read_csv(src)
    if df.empty:
        return {"items": [], "unit": "vessels", "granularity": granularity}
    if "time_bin" not in df.columns:
        raise HTTPException(status_code=500, detail="congestion_curve.csv missing time_bin")
    work = df.copy()
    work["time_bin"] = pd.to_datetime(work["time_bin"], errors="coerce")
    work = work.dropna(subset=["time_bin"])
    for col in ("presence_mmsi", "idle_mmsi"):
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    if granularity == "hour":
        out = work.sort_values("time_bin")[["time_bin", "presence_mmsi", "idle_mmsi"]]
        out["time_bin"] = out["time_bin"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return {"items": out.to_dict(orient="records"), "unit": "vessels", "granularity": "hour"}

    out = (
        work.assign(date=work["time_bin"].dt.floor("D"))
        .groupby("date", as_index=False)[["presence_mmsi", "idle_mmsi"]]
        .mean()
        .sort_values("date")
    )
    out["time_bin"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["date"])
    return {"items": out.to_dict(orient="records"), "unit": "vessels", "granularity": "day"}


@app.get("/api/series/waiting/day")
def series_waiting_day() -> Dict[str, Any]:
    src = METRICS_DIR / "waiting_time_by_day.csv"
    df = _read_csv(src)
    if df.empty:
        return {"items": [], "unit": "minutes"}
    for c in ("mean", "median", "p90", "p95", "max", "count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("mean", "median", "p90", "p95", "max"):
        if c in df.columns:
            df[f"{c}_hours"] = (df[c] / 60.0).round(2)
    return {"items": _as_records(df, "date"), "unit": "minutes"}


@app.get("/api/series/waiting/summary")
def series_waiting_summary() -> Dict[str, Any]:
    src = METRICS_DIR / "waiting_time_summary.csv"
    df = _read_csv(src)
    if df.empty:
        return {
            "count": 0,
            "mean_min": 0,
            "p90_min": 0,
            "p95_min": 0,
            "mean_hr": 0,
            "p90_hr": 0,
            "p95_hr": 0,
            "explain": {
                "p90": "P90 means 90% of waiting events are shorter than this value.",
                "p95": "P95 means 95% of waiting events are shorter than this value.",
            },
        }
    row = df.iloc[0].to_dict()
    mean_min = _safe_float(row.get("mean"))
    p90_min = _safe_float(row.get("p90"))
    p95_min = _safe_float(row.get("p95"))
    return {
        "count": int(_safe_float(row.get("count"))),
        "mean_min": round(mean_min, 2),
        "p90_min": round(p90_min, 2),
        "p95_min": round(p95_min, 2),
        "mean_hr": round(mean_min / 60.0, 2),
        "p90_hr": round(p90_min / 60.0, 2),
        "p95_hr": round(p95_min / 60.0, 2),
        "explain": {
            "p90": "P90 means 90% of waiting events are shorter than this value.",
            "p95": "P95 means 95% of waiting events are shorter than this value.",
        },
    }


def _ensure_congestion_forecast(horizon: int) -> Path:
    out = METRICS_DIR / "congestion_forecast.csv"
    src = METRICS_DIR / "congestion_curve.csv"
    if not src.exists() or _read_csv(src).empty:
        return out
    need_build = True
    if out.exists():
        df = _read_csv(out)
        need_build = df.empty or len(df) < horizon
    if need_build:
        _run_script(["scripts/run_forecast.py", "--horizon", str(horizon)])
    return out


def _ensure_waiting_forecast(horizon: int) -> Path:
    out = METRICS_DIR / "waiting_forecast.csv"
    src = METRICS_DIR / "waiting_time_by_day.csv"
    if not src.exists() or _read_csv(src).empty:
        return out
    need_build = True
    if out.exists():
        df = _read_csv(out)
        need_build = df.empty or len(df) < horizon
    if need_build:
        _run_script(["scripts/run_waiting_forecast.py", "--horizon", str(horizon)])
    return out


def _ensure_vision_forecast(horizon: int) -> Path:
    out = METRICS_DIR / "vision_forecast.csv"
    yolo_src = METRICS_DIR / "yolo_observed.csv"
    if not yolo_src.exists() or _read_csv(yolo_src).empty:
        return out
    need_build = True
    if out.exists():
        df = _read_csv(out)
        need_build = df.empty or len(df) < horizon
    if need_build:
        _run_script(
            [
                "scripts/run_vision_forecast.py",
                "--horizon",
                str(horizon),
                "--allow-missing-ais",
            ]
        )
    return out


@app.get("/api/forecast/congestion")
def forecast_congestion(horizon_days: int = 24, date: Optional[str] = None) -> Dict[str, Any]:
    horizon_days = max(1, min(horizon_days, 24))
    try:
        src = _ensure_congestion_forecast(horizon_days)
    except Exception:
        return {"items": [], "selected": None, "unit": "vessels"}
    if not src.exists():
        return {"items": [], "selected": None, "unit": "vessels"}
    df = _read_csv(src)
    if df.empty:
        return {"items": [], "selected": None, "unit": "vessels"}
    if "time_bin" not in df.columns:
        raise HTTPException(status_code=500, detail="congestion_forecast.csv missing time_bin")
    df = df.head(horizon_days).copy()
    if "forecast_value" in df.columns:
        df["forecast_value"] = pd.to_numeric(df["forecast_value"], errors="coerce").round(2)
    items = df.to_dict(orient="records")
    selected = None
    if date:
        hit = next((x for x in items if str(x.get("time_bin")) == date), None)
        if hit:
            selected = hit
    return {"items": items, "selected": selected, "unit": "vessels"}


@app.get("/api/forecast/waiting")
def forecast_waiting(horizon_days: int = 24, date: Optional[str] = None) -> Dict[str, Any]:
    horizon_days = max(1, min(horizon_days, 24))
    try:
        src = _ensure_waiting_forecast(horizon_days)
    except Exception:
        return {"items": [], "selected": None, "unit": "minutes"}
    if not src.exists():
        return {"items": [], "selected": None, "unit": "minutes"}
    df = _read_csv(src)
    if df.empty:
        return {"items": [], "selected": None, "unit": "minutes"}
    df = df.head(horizon_days).copy()
    for c in ("mean_forecast", "p90_forecast"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
            df[f"{c.replace('_forecast', '_hours')}"] = (df[c] / 60.0).round(2)
    items = df.to_dict(orient="records")
    selected = None
    if date:
        hit = next((x for x in items if str(x.get("date")) == date), None)
        if hit:
            selected = hit
    return {
        "items": items,
        "selected": selected,
        "unit": "minutes",
        "explain": {
            "p90": "P90 means 90% of predicted waiting events are below this value.",
            "p95": "P95 is currently shown in historical summary, not in forecast.",
        },
    }


@app.get("/api/forecast/vision")
def forecast_vision(horizon_days: int = 24, date: Optional[str] = None) -> Dict[str, Any]:
    horizon_days = max(1, min(horizon_days, 24))
    try:
        src = _ensure_vision_forecast(horizon_days)
    except Exception:
        # Vision forecast depends on YOLO inputs; return empty if unavailable.
        return {"items": [], "selected": None, "unit": "vessels"}
    df = _read_csv(src)
    if df.empty:
        return {"items": [], "selected": None, "unit": "vessels"}
    df = df.head(horizon_days).copy()
    for c in ("vision_forecast", "yolo_ship_eq_forecast", "ais_forecast"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    items = df.to_dict(orient="records")
    selected = None
    if date:
        hit = next((x for x in items if str(x.get("time_bin")) == date), None)
        if hit:
            selected = hit
    return {"items": items, "selected": selected, "unit": "vessels"}


@app.get("/api/evidence/cards")
def evidence_cards(limit: int = 20) -> Dict[str, Any]:
    files = sorted(EVIDENCE_DIR.glob("*.json"))
    if not files:
        return {"items": []}
    rows = []
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_file"] = str(fp.relative_to(ROOT))
        payload["_mtime"] = _mtime_iso(fp)
        rows.append(payload)
    rows.sort(key=lambda x: x.get("t_anchor", ""), reverse=True)
    return {"items": rows[: max(1, min(limit, 100))]}


@app.get("/api/map/ais-points")
def map_ais_points(limit: int = 3000) -> Dict[str, Any]:
    limit = max(100, min(limit, 10000))
    src = _find_ais_source_csv()
    if not src:
        return {"type": "FeatureCollection", "features": [], "source": None}
    chunks = pd.read_csv(src, chunksize=50000)
    rows: List[Dict[str, Any]] = []
    for ck in chunks:
        if "lon" not in ck.columns or "lat" not in ck.columns:
            continue
        sub = ck[["lon", "lat"] + [c for c in ("mmsi", "postime", "sog") if c in ck.columns]].copy()
        sub["lon"] = pd.to_numeric(sub["lon"], errors="coerce")
        sub["lat"] = pd.to_numeric(sub["lat"], errors="coerce")
        sub = sub.dropna(subset=["lon", "lat"])
        rows.extend(sub.to_dict(orient="records"))
        if len(rows) >= limit:
            break
    if len(rows) > limit:
        step = max(len(rows) // limit, 1)
        rows = rows[::step][:limit]
    features = []
    for r in rows:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
                "properties": {
                    "mmsi": str(r.get("mmsi", "")),
                    "postime": str(r.get("postime", "")),
                    "sog": _safe_float(r.get("sog"), 0.0),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features, "source": str(src.relative_to(ROOT))}


@app.post("/api/jobs/download-ais/start")
def start_download_ais(req: StartDownloadRequest) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "download_ais_pipeline.py",
        "--start",
        req.start,
        "--end",
        req.end,
        "--bbox",
        req.bbox,
        "--slice-hours",
        str(req.slice_hours),
        "--page-size",
        str(req.page_size),
        "--slice-threshold",
        str(req.slice_threshold),
    ]
    state = _start_job("download_ais", cmd, LOG_DIR / "download_ais.log")
    return {"job": state.model_dump(), "progress": _scan_latest_ais_file()}


@app.post("/api/jobs/pipeline/start")
def start_pipeline(req: StartPipelineRequest) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--horizon-days",
        str(req.horizon_days),
    ]
    state = _start_job("pipeline", cmd, LOG_DIR / "pipeline.log")
    return {"job": state.model_dump()}


@app.get("/api/jobs/status")
def jobs_status() -> Dict[str, Any]:
    with _jobs_lock:
        jobs = {k: v.model_dump() for k, v in _jobs.items()}
    for name, row in jobs.items():
        lf = Path(row.get("log_file")) if row.get("log_file") else None
        row["log_tail"] = _tail(lf) if lf else []
    return {
        "jobs": jobs,
        "download_progress": _scan_latest_ais_file(),
    }


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> Response:
    # Keep API/docs/openapi routes as backend-only.
    if full_path.startswith("api/") or full_path in {"api", "docs", "openapi.json", "redoc"}:
        raise HTTPException(status_code=404, detail="Not Found")

    asset = _frontend_file_or_none(full_path)
    if asset:
        return FileResponse(asset)

    index = _frontend_index()
    if index.exists():
        return FileResponse(index)

    raise HTTPException(status_code=404, detail="Frontend build not found; deploy frontend/dist or open /docs")
