from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import backend.app as api


def _patch_paths(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    metrics = root / "outputs" / "metrics"
    evidence = root / "outputs" / "evidence_cards"
    logs = root / "outputs" / "logs"
    raw_ais = root / "data" / "raw" / "ais"
    processed_ais = root / "data" / "processed" / "ais_cleaned"
    interim_ais = root / "data" / "interim" / "ais_clean"
    for p in [metrics, evidence, logs, raw_ais / "raw_tracks_csv", raw_ais / "raw_tracks", processed_ais, interim_ais]:
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "METRICS_DIR", metrics)
    monkeypatch.setattr(api, "EVIDENCE_DIR", evidence)
    monkeypatch.setattr(api, "LOG_DIR", logs)
    monkeypatch.setattr(api, "RAW_AIS_DIR", raw_ais)
    monkeypatch.setattr(api, "PROCESSED_AIS_DIR", processed_ais)
    monkeypatch.setattr(api, "INTERIM_AIS_DIR", interim_ais)
    api._cache_store.clear()


def test_health_ok_with_credentials_and_model(tmp_path: Path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    (api.METRICS_DIR / "congestion_curve.csv").write_text("time_bin,presence_mmsi,idle_mmsi\n2024-04-01,1,1\n", encoding="utf-8")
    (api.RAW_AIS_DIR / "raw_tracks_csv" / "sample.csv").write_text("mmsi,postime,lat,lon,sog\n1,2024-04-01,30.6,122.0,0.1\n", encoding="utf-8")
    model = api.ROOT / "assets" / "models"
    model.mkdir(parents=True, exist_ok=True)
    (model / "sar_ship_yolov8n.pt").write_text("x", encoding="utf-8")

    monkeypatch.setenv("MYVESSEL_CLIENT_ID", "id")
    monkeypatch.setenv("MYVESSEL_CLIENT_SECRET", "secret")
    monkeypatch.setenv("COPERNICUS_USER", "u")
    monkeypatch.setenv("COPERNICUS_PASSWORD", "p")

    client = TestClient(api.app)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["status"] == "ok"
    assert body["checks"]["pipeline_ready"]["has_ais_any"] is True
    assert body["checks"]["pipeline_ready"]["has_yolo_model"] is True
    assert body["checks"]["data"]["metrics"]["congestion_curve"]["exists"] is True


def test_health_degraded_without_credentials(tmp_path: Path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.delenv("MYVESSEL_CLIENT_ID", raising=False)
    monkeypatch.delenv("MYVESSEL_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("COPERNICUS_USER", raising=False)
    monkeypatch.delenv("COPERNICUS_PASSWORD", raising=False)

    client = TestClient(api.app)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["status"] == "degraded"
    assert any("credentials" in w.lower() for w in body["warnings"])


def test_map_ais_points_ignores_non_geo_csv(tmp_path: Path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    # Non-geo CSV should not be selected as map source.
    (api.RAW_AIS_DIR / "mmsi.csv").write_text("mmsi,count\n1,10\n", encoding="utf-8")
    # Real track CSV under raw_tracks_csv should be used.
    sub = api.RAW_AIS_DIR / "raw_tracks_csv" / "100"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "slice.csv").write_text(
        "mmsi,postime,lat,lon,sog\n100,2024-04-01 00:00:00,30.60,122.00,0.2\n",
        encoding="utf-8",
    )

    client = TestClient(api.app)
    res = client.get("/api/map/ais-points?limit=100")
    assert res.status_code == 200
    body = res.json()
    assert body["source"].endswith("data\\raw\\ais\\raw_tracks_csv\\100\\slice.csv")
    assert len(body["features"]) == 1


def test_forecast_vision_returns_semantic_unit_and_confidence(tmp_path: Path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    vf = api.METRICS_DIR / "vision_forecast.csv"
    vf.write_text(
        "\n".join(
            [
                "time_bin,vision_forecast,yolo_ship_eq_forecast,mode,semantic_unit,confidence_level,confidence_reason",
                "2024-07-01,20,20,yolo_only,detection_index,low,YOLO-only forecast without AIS calibration.",
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(api.app)
    res = client.get("/api/forecast/vision?horizon_days=3")
    assert res.status_code == 200
    body = res.json()
    assert body["unit"] == "detection_index"
    assert body["confidence"]["level"] == "low"
    assert "trend signals" in body["business_note"]
    assert len(body["items"]) == 1


def test_meta_uses_ttl_cache(tmp_path: Path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    calls = {"n": 0}

    def _fake_meta():
        calls["n"] += 1
        return {"server_time": "x", "last_updated_utc": None, "files": {}, "ais_download": {}, "active_yolo_model": "m"}

    monkeypatch.setattr(api, "_build_meta_payload", _fake_meta)
    monkeypatch.setattr(api, "DEFAULT_META_CACHE_TTL_SEC", 60.0)

    client = TestClient(api.app)
    r1 = client.get("/api/meta")
    r2 = client.get("/api/meta")
    assert r1.status_code == 200 and r2.status_code == 200
    assert calls["n"] == 1


def test_health_uses_ttl_cache(tmp_path: Path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    calls = {"n": 0}

    def _fake_health():
        calls["n"] += 1
        return {"ok": True, "status": "ok", "warnings": [], "checks": {}}

    monkeypatch.setattr(api, "_health_checks", _fake_health)
    monkeypatch.setattr(api, "DEFAULT_HEALTH_CACHE_TTL_SEC", 60.0)

    client = TestClient(api.app)
    r1 = client.get("/api/health")
    r2 = client.get("/api/health")
    assert r1.status_code == 200 and r2.status_code == 200
    assert calls["n"] == 1
