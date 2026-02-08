from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI = {
    "id": "yangshan_anchor",
    "bbox": {"lon_min": 121.90, "lon_max": 122.25, "lat_min": 30.50, "lat_max": 30.75},
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _pick_anchors(curve: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if curve.empty or "time_bin" not in curve.columns:
        return pd.DataFrame()
    work = curve.copy()
    work["time_bin"] = pd.to_datetime(work["time_bin"], errors="coerce")
    work = work.dropna(subset=["time_bin"])
    for c in ("idle_mmsi", "presence_mmsi"):
        if c not in work.columns:
            work[c] = 0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    peaks = work.sort_values(["idle_mmsi", "presence_mmsi"], ascending=[False, False]).head(max(1, top_n))
    return peaks


def _dwell_examples(events: pd.DataFrame, anchor_day: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if events.empty:
        return []
    e = events.copy()
    if "end_time" not in e.columns or "duration_min" not in e.columns:
        return []
    e["end_time"] = pd.to_datetime(e["end_time"], errors="coerce")
    e = e.dropna(subset=["end_time"])
    e["day"] = e["end_time"].dt.strftime("%Y-%m-%d")
    e["duration_min"] = pd.to_numeric(e["duration_min"], errors="coerce").fillna(0)
    day_rows = e[e["day"] == anchor_day].sort_values("duration_min", ascending=False).head(top_k)
    out: List[Dict[str, Any]] = []
    for _, r in day_rows.iterrows():
        out.append(
            {
                "mmsi": str(r.get("mmsi", "")),
                "start_time": str(r.get("start_time", "")),
                "end_time": str(r.get("end_time", "")),
                "dwell_min": float(r.get("duration_min", 0)),
                "confidence": "medium",
            }
        )
    return out


def _yolo_summary(yolo: pd.DataFrame, anchor_day: str) -> Dict[str, Any]:
    if yolo.empty or "time_bin" not in yolo.columns:
        return {"available": False}
    y = yolo.copy()
    y["time_bin"] = pd.to_datetime(y["time_bin"], errors="coerce")
    y = y.dropna(subset=["time_bin"])
    y["day"] = y["time_bin"].dt.strftime("%Y-%m-%d")
    day = y[y["day"] == anchor_day]
    if day.empty:
        return {"available": False}
    det = pd.to_numeric(day.get("yolo_detections", 0), errors="coerce").fillna(0)
    conf = pd.to_numeric(day.get("yolo_mean_conf", 0), errors="coerce").fillna(0)
    return {
        "available": True,
        "detections": int(det.sum()),
        "mean_confidence": round(float(conf.mean()), 3),
        "files": int(pd.to_numeric(day.get("yolo_files", 0), errors="coerce").fillna(0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence cards from metrics")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--out-dir", default="outputs/evidence_cards")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    metrics_dir = ROOT / args.metrics_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_csv(metrics_dir / "congestion_curve.csv")
    waiting_day = _load_csv(metrics_dir / "waiting_time_by_day.csv")
    waiting_events = _load_csv(metrics_dir / "waiting_time_events.csv")
    yolo = _load_csv(metrics_dir / "yolo_observed.csv")

    peaks = _pick_anchors(curve, args.top_n)
    if peaks.empty:
        index_path = out_dir / "index.csv"
        pd.DataFrame(columns=["card_id", "t_anchor", "peak_rank", "file"]).to_csv(index_path, index=False, encoding="utf-8-sig")
        print(f"[done] no peaks found, empty index -> {index_path}")
        return

    idx_rows: List[Dict[str, Any]] = []
    for rank, (_, r) in enumerate(peaks.iterrows(), start=1):
        t_anchor = pd.to_datetime(r["time_bin"])
        day = t_anchor.strftime("%Y-%m-%d")
        t_text = t_anchor.strftime("%Y-%m-%d %H:%M:%S")
        card_id = f"card_{t_anchor.strftime('%Y%m%dT%H%M%S')}"

        day_wait = waiting_day[waiting_day.get("date", "") == day] if not waiting_day.empty else pd.DataFrame()
        wait_mean = float(pd.to_numeric(day_wait.get("mean", 0), errors="coerce").fillna(0).mean()) if not day_wait.empty else 0.0
        wait_p90 = float(pd.to_numeric(day_wait.get("p90", 0), errors="coerce").fillna(0).mean()) if not day_wait.empty else 0.0

        yolo_info = _yolo_summary(yolo, day)
        card = {
            "card_id": card_id,
            "port_id": "yangshan",
            "roi_id": DEFAULT_ROI["id"],
            "roi_bbox": DEFAULT_ROI["bbox"],
            "t_anchor": t_text,
            "time_window": {"start": f"{day} 00:00:00", "end": f"{day} 23:59:59"},
            "congestion_snapshot": {
                "presence_count": float(r.get("presence_mmsi", 0)),
                "idle_count": float(r.get("idle_mmsi", 0)),
                "calibrated_count": None,
                "confidence_interval": None,
                "data_coverage": "default_coverage_unknown",
            },
            "ais_evidence": {
                "ais_tracks_geojson": "outputs/export/ais_points_map.html",
                "dwell_examples": _dwell_examples(waiting_events, day, top_k=3),
                "waiting_mean_min": round(wait_mean, 2),
                "waiting_p90_min": round(wait_p90, 2),
            },
            "satellite_yolo_evidence": {
                "sensor": "Sentinel-1 GRD",
                "summary": yolo_info,
                "match_stats": {"A_satellite_only": None, "B_ais_only": None, "AB_matched": None},
            },
            "explanation": (
                f"At {t_text} UTC, congestion is high with presence={int(float(r.get('presence_mmsi', 0)))} "
                f"and idle={int(float(r.get('idle_mmsi', 0)))} vessels. "
                f"Waiting metrics on {day} show mean={wait_mean:.1f} min and p90={wait_p90:.1f} min. "
                "Waiting uses ROI-in + low-speed (SOG<=0.5 kn), with max gap 60 min and min duration 20 min. "
                "Large AIS gaps can increase uncertainty in dwell estimates."
            ),
        }

        out_file = out_dir / f"{card_id}.json"
        out_file.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        idx_rows.append({"card_id": card_id, "t_anchor": t_text, "peak_rank": rank, "file": str(out_file)})

    index_path = out_dir / "index.csv"
    pd.DataFrame(idx_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    print(f"[done] cards={len(idx_rows)} index={index_path}")


if __name__ == "__main__":
    main()

