# YOLOv11 Logo Detector (Nuclio for CVAT)

This function serves an Ultralytics YOLOv11 model for logo detection via Nuclio, returning CVAT-compatible rectangles.

## Files
- `function.yaml`: Nuclio spec (GPU optional)
- `main.py`: Inference handler using `ultralytics` YOLO
- `scripts/download_model.sh`: Helper to fetch a YOLOv11 weights file

## Model
Place your weights at `yolo11-logo.pt` next to `function.yaml`, or set env `MODEL_URL` to download at deploy time.

## Deploy
```
nuctl deploy yolov11-logo -p . -n cvat
```

## Request
POST an image (binary, multipart `image`, or JSON with `image` base64/URL). Returns JSON rectangles.


