import io
import os
import json
import time
import base64
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image, ImageOps

import torch
from ultralytics import YOLO


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


def _pil_from_bytes(b: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB")


def _maybe_b64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return b""


_MODEL: YOLO = None
_DEVICE: str = "cpu"


def init_context(context):
    global _MODEL, _DEVICE
    use_gpu = _env("USE_GPU", "1") == "1"
    _DEVICE = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    torch.set_grad_enabled(False)

    model_path = _env("MODEL_PATH", "/opt/nuclio/yolo11-logo.pt")
    model_url = _env("MODEL_URL", "")

    if not os.path.isfile(model_path):
        if model_url:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            context.logger.info(f"Downloading model from {model_url}")
            r = requests.get(model_url, timeout=300)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)
        else:
            # fallback to built-in ultralytics weights if path missing
            context.logger.warn("MODEL_PATH missing; using ultralytics default 'yolo11n.pt'")
            model_path = "yolo11n.pt"

    t0 = time.time()
    _MODEL = YOLO(model_path)
    _MODEL.to(_DEVICE)
    context.logger.info(
        f"YOLOv11 loaded: weights={model_path} device={_DEVICE} in {time.time()-t0:.2f}s"
    )
    try:
        names = _MODEL.names if hasattr(_MODEL, "names") else {}
        context.logger.info(f"model classes: {len(names)} -> {list(names.values())[:10]}")
    except Exception:
        pass


def _read_image_from_event(event) -> Image.Image:
    ct = (event.content_type or "").lower()
    if ct.startswith("image/") or "application/octet-stream" in ct:
        return _pil_from_bytes(event.body)

    if "application/json" in ct or ct == "":
        body = event.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8") if body else "{}"
        data = body if isinstance(body, dict) else json.loads(body or "{}")
        im = data.get("image")
        if isinstance(im, str):
            if im.startswith("http://") or im.startswith("https://"):
                r = requests.get(im, timeout=60)
                r.raise_for_status()
                return _pil_from_bytes(r.content)
            b = _maybe_b64(im)
            if b:
                return _pil_from_bytes(b)
        url = data.get("image_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return _pil_from_bytes(r.content)
        raise ValueError("JSON must include 'image' or 'image_url'")

    if "multipart/form-data" in ct:
        files = getattr(event, "files", {}) or {}
        if isinstance(files, dict) and "image" in files:
            f = files["image"]
            data = f.get("data") if isinstance(f, dict) else None
            if isinstance(data, (bytes, bytearray)) and data:
                return _pil_from_bytes(data)
        raise ValueError("multipart/form-data requires file field 'image'")

    raise ValueError(f"Unsupported Content-Type: {ct}")


def handler(context, event):
    try:
        conf_thr = float(_env("CONF_THRESHOLD", "0.25"))
        iou_thr = float(_env("IOU_THRESHOLD", "0.45"))
        output_label = _env("OUTPUT_LABEL", "logo")
        filter_classes = _env("FILTER_CLASSES", "")
        use_model_labels = _env("USE_MODEL_LABELS", "0") == "1"

        # Request logging
        ct = (event.content_type or "")
        body_len = len(event.body) if isinstance(event.body, (bytes, bytearray)) else -1
        try:
            hdrs = getattr(event, "headers", {}) or {}
            hdr_keys = list(hdrs.keys())[:12]
        except Exception:
            hdr_keys = []
        context.logger.info(
            f"invoke ct={ct!r} body_len={body_len} headers_keys_preview={hdr_keys}"
        )
        context.logger.info(
            f"config conf={conf_thr} iou={iou_thr} output_label={output_label} use_model_labels={use_model_labels}"
        )

        image = _read_image_from_event(event)
        context.logger.info(f"loaded image size={image.size}")

        t0 = time.time()
        # Run YOLOv11 predict
        results = _MODEL.predict(
            source=np.array(image),
            device=_DEVICE,
            conf=conf_thr,
            iou=iou_thr,
            verbose=False,
        )
        infer_ms = (time.time() - t0) * 1000.0
        context.logger.info(f"inference time={infer_ms:.1f}ms batches={len(results)}")

        dets: List[Dict[str, Any]] = []
        names = _MODEL.names if hasattr(_MODEL, "names") else {}
        class_filter: List[int] = []
        if filter_classes.strip():
            tokens = [t.strip() for t in filter_classes.split(",") if t.strip()]
            for t in tokens:
                if t.isdigit():
                    class_filter.append(int(t))
                else:
                    # name to id
                    for k, v in names.items():
                        if str(v) == t:
                            class_filter.append(int(k))
        if class_filter:
            context.logger.info(f"class filter active: {class_filter}")

        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()  # Nx4
            confs = r.boxes.conf.cpu().numpy()  # N
            cls = r.boxes.cls.cpu().numpy().astype(int)  # N
            context.logger.info(
                f"pred batch: raw_dets={len(xyxy)} image_shape={getattr(r, 'orig_shape', None)}"
            )
            for i in range(len(xyxy)):
                cid = int(cls[i])
                if class_filter and cid not in class_filter:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                if use_model_labels:
                    label_name = str(names.get(cid, output_label))
                else:
                    label_name = output_label
                dets.append(
                    {
                        "label": label_name,
                        "type": "rectangle",
                        "points": [x1, y1, x2, y2],
                        "confidence": float(confs[i]),
                    }
                )

        # Preview detections
        if not dets:
            context.logger.info("returning 0 detections")
        else:
            context.logger.info(f"returning {len(dets)} detections")
            for idx, d in enumerate(dets[: min(5, len(dets))]):
                pts = d.get("points", [])
                context.logger.info(
                    f"  det[{idx}] label={d.get('label')} conf={d.get('confidence'):.3f}"
                    f" box=[{pts[0]:.1f},{pts[1]:.1f},{pts[2]:.1f},{pts[3]:.1f}]"
                )

        return context.Response(
            body=json.dumps(dets, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
            status_code=200,
        )

    except Exception as e:
        try:
            context.logger.error(f"inference error: {e}")
            try:
                context.logger.error(traceback.format_exc())
            except Exception:
                pass
        except Exception:
            pass
        return context.Response(
            body=json.dumps({"error": str(e)}, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
            status_code=500,
        )


