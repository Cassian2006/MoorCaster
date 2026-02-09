@echo off
setlocal
cd /d %~dp0\..

python scripts\train_lssdd_gpu_live.py ^
  --data-yaml data/interim/lssdd_yolo/data.yaml ^
  --base-model assets/models/moorcaster_ship_lssdd_smoke.pt ^
  --epochs 30 ^
  --imgsz 800 ^
  --batch 8 ^
  --device 0 ^
  --workers 4 ^
  --patience 10 ^
  --project outputs/train ^
  --run-name lssdd_full_30ep_gpu ^
  --export-model assets/models/moorcaster_ship_lssdd.pt ^
  --log-file outputs/logs/train_lssdd_gpu_live.log ^
  --heartbeat-sec 20

endlocal
