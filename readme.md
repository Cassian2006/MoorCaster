# MoorCaster (Rebuild)

本仓库用于洋山锚地 `Congestion & Waiting` 分析，当前按以下固定口径运行：

- ROI (唯一 bbox): `lon 121.90~122.25, lat 30.50~30.75`
- 历史时间窗: `2024-04-01 00:00:00` 到 `2024-06-30 23:59:59`
- 在场船舶: `ROI 内`
- 低速停留: `ROI 内且 sog <= 0.5 kn`
- 等待事件阈值: `min_duration=20 min, max_gap=60 min`
- 预测窗口: `未来 24 天（可选任意一天）`

## 1) 环境准备

```powershell
cd "C:\Users\cai yuan qi\Desktop\MoorCaster"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

YOLO / S1 相关依赖（可选）：

```powershell
pip install -r requirements-vision.txt
```

设置 API 凭证（仅本地环境变量）：

```powershell
$env:MYVESSEL_CLIENT_ID="your_id"
$env:MYVESSEL_CLIENT_SECRET="your_secret"
```

## 2) AIS 下载（支持断点续跑 + 进度打印）

主入口：

```powershell
python .\download_ais_pipeline.py
```

自定义参数示例：

```powershell
python .\download_ais_pipeline.py `
  --start "2024-04-01 00:00:00" `
  --end "2024-06-30 23:59:59" `
  --bbox "121.90,30.75,122.25,30.50" `
  --slice-hours 6 `
  --page-size 100 `
  --slice-threshold 9000 `
  --output-format both
```

下载产物默认写入：

- `data/raw/ais/raw_tracks/` (`jsonl`)
- `data/raw/ais/raw_tracks_csv/` (`csv`)
- `data/raw/ais/mmsi.csv`

## 3) 数据处理与预测

### 3.1 纯 AIS 流程

```powershell
python .\src\data\clip_ais.py
python .\src\data\ais_clean_basic.py
python .\scripts\run_metrics.py
python .\scripts\run_forecast.py --horizon 24
python .\scripts\run_waiting_forecast.py --horizon 24
python .\scripts\build_evidence_cards.py
python .\scripts\export_map.py
```

### 3.2 一键流程

```powershell
python .\scripts\run_pipeline.py --horizon-days 24
```

## 4) S1 + YOLO（视觉证据）

### 4.1 拉取 S1 STAC 清单/quicklook

```powershell
python .\scripts\download_s1_stac.py --download
```

### 4.2 自动视觉流程

```powershell
python .\scripts\run_s1_yolo_auto.py --horizon 24
```

视觉相关输出：

- YOLO 检测：`outputs/yolo/*.json`
- 日汇总：`outputs/metrics/yolo_observed.csv`
- 视觉融合预测：`outputs/metrics/vision_forecast.csv`

## 5) 启动后端与前端

后端（FastAPI）：

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

前端（Vite + React）：

```powershell
cd .\frontend
npm install
npm run dev
```

访问：

- 前端：`http://127.0.0.1:5173`
- 后端健康检查：`http://127.0.0.1:8000/api/health`

## 6) 前端已提供的能力

- 中英文切换
- 最后更新时间显示（来自 `/api/meta`）
- 总览：地图 + 拥堵曲线 + 等待曲线（含单位）
- 预测：未来 24 天任意日期选择
- P90/P95 解释文案
- 证据卡弹窗查看
- 任务页：可直接触发下载与 pipeline，并看日志和文件增长

## 7) 关键输出路径

- `outputs/metrics/congestion_curve.csv`
- `outputs/metrics/waiting_time_by_day.csv`
- `outputs/metrics/waiting_time_summary.csv`
- `outputs/metrics/congestion_forecast.csv`
- `outputs/metrics/waiting_forecast.csv`
- `outputs/metrics/vision_forecast.csv`
- `outputs/evidence_cards/*.json`
- `outputs/export/ais_points_map.html`


## 8) Auto Handoff (AIS + S1 -> Pipeline)

When AIS and S1 are ready, run this and it will wait, then trigger full pipeline automatically:

```powershell
python .\scripts\wait_and_run_pipeline.py --ais-min-files 1 --s1-min-files 5 --horizon-days 24 --yolo-model yolov8n.pt
```

Useful logs:

- AIS download: `outputs/logs/download_ais.log`
- Auto handoff: `outputs/logs/wait_pipeline.log`
- S1 download: `outputs/logs/download_s1.log`

## 9) SAR YOLO Model (New)

Train SAR-specific YOLO weights (GPU recommended):

```powershell
python .\scripts\train_sar_yolo_model.py --epochs 20 --imgsz 640 --batch 16 --device 0 --workers 4
```

Output model:

- `assets/models/sar_ship_yolov8n.pt`

After model exists, `run_s1_yolo_auto.py` and `run_pipeline.py` will prefer this model automatically.

## 10) S1 High-Quality (GRD Product ZIP, New)

Use this mode when you want YOLO to run on higher-quality S1 source (not quicklook thumbnails).

Set Copernicus credentials first:

```powershell
$env:COPERNICUS_USER="your_email"
$env:COPERNICUS_PASSWORD="your_password"
```

Download S1 product ZIP with resume + progress:

```powershell
python .\scripts\download_s1_stac.py `
  --asset product `
  --download `
  --resume `
  --max-items 1200 `
  --start "2024-04-01T00:00:00Z" `
  --end "2024-06-30T23:59:59Z"
```

Prepare YOLO-ready PNG from GRD ZIP:

```powershell
python .\scripts\prepare_s1_grd.py `
  --input-dir data/raw/s1/grd_zip `
  --output-dir data/interim/s1_grd_png `
  --prefer-pol vv
```

Then run YOLO auto pipeline (it will prefer `data/interim/s1_grd_png` automatically):

```powershell
python .\scripts\run_s1_yolo_auto.py --horizon 24
```

## 11) Frontend (Figma Theme Integration, New)

- Frontend now uses Figma exported theme tokens from `frontend/styles/`.
- Header shows:
- `Data Last Updated (UTC)` from `/api/meta`
- `Frontend Build Time (UTC)` injected at build time

## 12) Testing (Pytest, New)

Run all tests:

```powershell
python -m pytest -q
```

Current test scope:

- Metrics logic: `tests/test_metrics_congestion_waiting.py`
- Forecast logic: `tests/test_metrics_forecast.py`
- API health checks: `tests/test_backend_api_health.py`

## 13) Health Checks (Enhanced)

Endpoint:

```text
GET /api/health
```

Returns:

- `ok` / `status` (`ok` or `degraded`)
- `warnings`
- `checks.runtime`
- `checks.credentials`
- `checks.filesystem`
- `checks.data` (AIS/S1/metrics counts and file status)
- `checks.models`
- `checks.activity` (recent log activity)
- `checks.pipeline_ready`

## 14) S1 Training Pipeline (New)

After S1 GRD ZIP download is done, you can build pseudo labels and fine-tune YOLO:

```powershell
python .\scripts\run_s1_training_pipeline.py `
  --s1-grd-zip-dir data/raw/s1/grd_zip `
  --s1-grd-png-dir data/interim/s1_grd_png `
  --yolo-output outputs/yolo `
  --dataset-dir data/interim/s1_yolo_pseudo `
  --epochs 20 `
  --imgsz 960 `
  --batch 8 `
  --device 0
```

What this command does:

- Prepare GRD ZIP -> PNG (`prepare_s1_grd.py`)
- Run YOLO on S1 PNG (`run_yolo.py`)
- Build pseudo-label YOLO dataset (`build_s1_pseudo_yolo_dataset.py`)
- Fine-tune model if labeled images >= threshold (`train_s1_yolo_finetune.py`)

## 15) Current Default Model (Prediction)

Current preferred YOLO weights order is:

1. `assets/models/moorcaster_ship_lssdd_recall_r1.pt`
2. `assets/models/moorcaster_ship_lssdd.pt`
3. `assets/models/sar_ship_yolov8n.pt`
4. `yolov8n.pt`

So once `moorcaster_ship_lssdd_recall_r1.pt` exists, pipeline forecast will use it by default.

## 16) User-Side Forecast Pipeline (No Download)

Web job button (`Refresh Forecast`) triggers local processing using existing downloaded data only:

- Read local AIS/S1 data
- Run YOLO inference with active model
- Rebuild `yolo_observed.csv`
- Recompute `vision_forecast.csv`
- Rebuild evidence cards and map export

Backend endpoint:

```text
POST /api/jobs/pipeline/start
```

Optional request body:

```json
{
  "horizon_days": 24,
  "yolo_model": "assets/models/moorcaster_ship_lssdd_recall_r1.pt"
}
```

Main outputs:

- `data/interim/s1_yolo_pseudo/data.yaml`
- `data/interim/s1_yolo_pseudo/summary.json`
- `assets/models/sar_ship_yolov8n_s1.pt`

## 12) YOLO Ship Labeling (Quick Start)

1) Prepare images to label (from S1 GRD PNG):

```powershell
python .\scripts\prepare_yolo_labeling.py --input-dir data\interim\s1_grd_png --output-dir data\annotations\ship\images_all --limit 0
```

2) Install label tool (one-time):

```powershell
pip install labelImg
```

3) Launch labelImg, set format to YOLO, then:
- Open Dir: `data\annotations\ship\images_all`
- Change Save Dir: `data\annotations\ship\labels_all`
- Create class: `ship`
- Draw boxes around ships and save

4) Split dataset into train/val:

```powershell
python .\scripts\split_yolo_dataset.py --images-dir data\annotations\ship\images_all --labels-dir data\annotations\ship\labels_all --out-dir data\annotations\ship --val-ratio 0.2
```

5) Training (optional, requires Ultralytics):

```powershell
pip install ultralytics

yolo detect train data=configs\yolo\ship.yaml model=yolov8n.pt epochs=30 imgsz=1024 batch=8
```

### LabelMe (recommended if labelImg crashes)

Start labelme:

```powershell
labelme
```

In labelme:
- Open Dir: `data\annotations\ship\images_all`
- Save Dir: `data\annotations\ship\labelme`
- Label name: `ship`
- Shape: rectangle (or polygon, we will convert to bbox)

Convert LabelMe JSON to YOLO txt:

```powershell
python .\scripts\labelme_to_yolo.py --input-dir data\annotations\ship\labelme --output-dir data\annotations\ship\labels_all --label ship
```
