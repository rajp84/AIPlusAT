import os
import io
import re
import json
import time
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from importlib import resources as importlib_resources

# GroundingDINO
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T

# =========================
# Globals
# =========================
_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", re.IGNORECASE)


# =========================
# Small helpers
# =========================
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


def _resolve_cfg_path() -> str:
    """Locate GroundingDINO_SwinT_OGC.py reliably."""
    cfg_env = _env("DINO_CFG")
    if cfg_env and os.path.isfile(cfg_env):
        return cfg_env

    try:
        return str(
            (importlib_resources.files("groundingdino") / "config" / "GroundingDINO_SwinT_OGC.py")
        )
    except Exception:
        pass

    candidates = [
        "/opt/conda/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "/usr/local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Cannot locate GroundingDINO_SwinT_OGC.py (set DINO_CFG to an absolute path)")


def _ensure_weights(context, dst_path: str) -> str:
    """Ensure weights exist at dst_path. Download if missing (primary + fallback)."""
    if os.path.isfile(dst_path):
        return dst_path

    primary = _env(
        "DINO_WEIGHTS_URL",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    )
    fallbacks = [
        "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    ]
    urls = [primary] + fallbacks

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    last_err = None
    for url in urls:
        try:
            context.logger.info(f"Downloading GroundingDINO weights from {url}")
            with requests.get(url, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(dst_path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            context.logger.info(f"Weights saved to {dst_path}")
            return dst_path
        except Exception as e:
            last_err = e
            context.logger.warn(f"Failed to fetch weights from {url}: {e}")

    raise RuntimeError(f"Could not download GroundingDINO weights to {dst_path}: {last_err}")


def _pil_from_bytes(b: bytes) -> Image.Image:
    # Apply EXIF orientation and ensure RGB
    return ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB")


def _bytes_from_possible_base64(s: str) -> bytes:
    m = _DATA_URL_RE.match(s)
    if m:
        return base64.b64decode(m.group("b64"))
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return b""


def _parse_labels_field(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        # try JSON array
        try:
            arr = json.loads(val)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        # comma/semicolon separated
        parts = [p.strip() for p in re.split(r"[;,]", val) if p.strip()]
        if parts:
            return parts
        if val.strip():
            return [val.strip()]
    return []


def _labels_from_headers(event) -> Tuple[List[str], Optional[str]]:
    """Try to harvest labels/prompt from headers, if present."""
    labels: List[str] = []
    prompt: Optional[str] = None
    try:
        hdrs = getattr(event, "headers", {}) or {}
        # normalize keys to lowercase strings
        norm = {}
        for k, v in hdrs.items():
            try:
                k = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            except Exception:
                k = str(k)
            norm[k.lower()] = v if isinstance(v, str) else (v.decode() if isinstance(v, (bytes, bytearray)) else str(v))

        for key in ("x-cvat-labels", "x-labels", "x-label"):
            if key in norm and norm[key]:
                try:
                    guess = json.loads(norm[key])
                    if isinstance(guess, list) and guess:
                        labels = _parse_labels_field(guess)
                        break
                except Exception:
                    vals = _parse_labels_field(norm[key])
                    if vals:
                        labels = vals
                        break

        for key in ("x-cvat-prompt", "x-prompt"):
            if key in norm and norm[key]:
                prompt = norm[key].strip()
                break
    except Exception:
        pass
    return labels, prompt


def _gdino_transform():
    # GroundingDINO default recipe: short side ~800, max 1333, normalize
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


def _prepare_image_for_model(img_pil: Image.Image):
    """Return (pil, tensor, (origW,origH), (procW,procH)) with proper preprocessing."""
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    W, H = img_pil.size
    transform = _gdino_transform()
    image_t, _ = transform(img_pil, None)   # CHW float32 tensor normalized
    procH, procW = image_t.shape[-2:]
    return img_pil, image_t, (W, H), (procW, procH)


# ---------- Box handling (robust to multiple formats) ----------
def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    cxy = boxes[:, 0:2]
    wh  = boxes[:, 2:4]
    x1y1 = cxy - wh / 2
    x2y2 = cxy + wh / 2
    return torch.cat([x1y1, x2y2], dim=1)

def _clamp_and_fix(xyxy: torch.Tensor, W: int, H: int) -> torch.Tensor:
    if xyxy.numel() == 0:
        return xyxy
    x1 = xyxy[:, 0].clamp_(0, W - 1)
    y1 = xyxy[:, 1].clamp_(0, H - 1)
    x2 = xyxy[:, 2].clamp_(0, W - 1)
    y2 = xyxy[:, 3].clamp_(0, H - 1)
    x_lo = torch.minimum(x1, x2)
    x_hi = torch.maximum(x1, x2)
    y_lo = torch.minimum(y1, y2)
    y_hi = torch.maximum(y1, y2)
    return torch.stack([x_lo, y_lo, x_hi, y_hi], dim=1)

def _validity_score(xyxy: torch.Tensor, W: int, H: int) -> float:
    """Higher is better: in-bounds + positive area."""
    if xyxy.numel() == 0:
        return 0.0
    x1, y1, x2, y2 = xyxy[:,0], xyxy[:,1], xyxy[:,2], xyxy[:,3]
    w = (x2 - x1).clamp_min(0)
    h = (y2 - y1).clamp_min(0)
    area_ok = ((w >= 2.0) & (h >= 2.0)).float()
    in_bounds = ((x1 >= -1) & (y1 >= -1) & (x2 <= W+1) & (y2 <= H+1)).float()
    return float((area_ok * in_bounds).mean().item())

def _boxes_to_xyxy_original(
    boxes: torch.Tensor, origW: int, origH: int, procW: int, procH: int, context=None
) -> torch.Tensor:
    """
    Robustly convert model outputs to ORIGINAL-pixel XYXY.
    Tries:
      - normalized CXCYWH -> XYXY (proc) -> scale to original
      - normalized XYXY   -> scale to original
      - processed-pixel XYXY -> scale to original
      - assume already original-pixel XYXY
    Picks the interpretation with best validity score when ambiguous.
    """
    if boxes.numel() == 0:
        if context: context.logger.info("bbx: empty")
        return boxes

    b = boxes.clone()
    bmax, bmin = float(b.max().item()), float(b.min().item())

    # normalized in [0,1]
    if bmax <= 1.00001 and bmin >= -0.00001:
        to_proc = torch.tensor([procW, procH, procW, procH], dtype=b.dtype, device=b.device)
        # A) cxcywh hypothesis
        xyxyA_proc = _cxcywh_to_xyxy(b) * to_proc
        # B) xyxy hypothesis
        xyxyB_proc = b * to_proc

        # scale to original
        sx, sy = (origW / float(procW)), (origH / float(procH))
        to_orig = torch.tensor([sx, sy, sx, sy], dtype=b.dtype, device=b.device)
        xyxyA_orig = xyxyA_proc * to_orig
        xyxyB_orig = xyxyB_proc * to_orig

        scoreA = _validity_score(xyxyA_orig, origW, origH)
        scoreB = _validity_score(xyxyB_orig, origW, origH)
        chosen = xyxyA_orig if scoreA >= scoreB else xyxyB_orig
        mode = "norm_cxcywh→proc→orig" if scoreA >= scoreB else "norm_xyxy→proc→orig"
        if context: context.logger.info(f"bbx mode={mode} scoreA={scoreA:.3f} scoreB={scoreB:.3f}")
        return _clamp_and_fix(chosen, origW, origH)

    # processed pixel space?
    if bmax <= max(procW, procH) + 1.0:
        sx, sy = (origW / float(procW)), (origH / float(procH))
        to_orig = torch.tensor([sx, sy, sx, sy], dtype=b.dtype, device=b.device)
        if context: context.logger.info("bbx mode=proc_xyxy→orig")
        return _clamp_and_fix(b * to_orig, origW, origH)

    # otherwise assume already original-pixel XYXY
    if context: context.logger.info("bbx mode=orig_xyxy")
    return _clamp_and_fix(b, origW, origH)
# --------------------------------------------------------------


def _build_caption_from_tokens(tokens: List[str]) -> str:
    toks = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
    return (" . ".join(toks) + " .") if toks else ""


def _labels_from_mapping(mapping: Any) -> List[str]:
    """
    CVAT AA sends mapping like: {"object": {"name": "face", "attributes": {...}}, ...}
    Extract the target label names.
    """
    out: List[str] = []
    if isinstance(mapping, dict):
        for v in mapping.values():
            if isinstance(v, dict):
                name = v.get("name")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
    return out


def _merge_prompt_hints_into_params(params: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Harvest labels/prompt from various common fields in CVAT payloads."""
    # labels
    for k in ("labels", "label"):
        if k in src and src[k] not in (None, ""):
            vals = _parse_labels_field(src[k]) if k == "labels" else [str(src[k]).strip()]
            if vals:
                params["labels"] = vals

    # mapping → labels
    if "mapping" in src and "labels" not in params:
        mapped = _labels_from_mapping(src["mapping"])
        if mapped:
            params["labels"] = mapped

    # prompt-like fields (include singular 'phrase' used by AA)
    prompt_candidates: List[str] = []
    for key in ("prompt", "text_prompt", "text", "query", "phrase"):
        v = src.get(key)
        if isinstance(v, str) and v.strip():
            prompt_candidates.append(v.strip())
    for key in ("queries", "phrases"):
        v = src.get(key)
        if isinstance(v, list):
            prompt_candidates.extend([str(x).strip() for x in v if str(x).strip()])

    if prompt_candidates:
        seen, ordered = set(), []
        for t in prompt_candidates:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        params["prompt"] = _build_caption_from_tokens(ordered)


def _build_caption_from_params(params: dict) -> str:
    """
    Priority:
      1) explicit prompt
      2) labels / label (build "a . b .")
      3) env DEFAULT_PROMPT
      4) fallback 'object .'
    """
    p = params.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()

    labels = params.get("labels")
    if isinstance(labels, list) and labels:
        return _build_caption_from_tokens(labels)

    envp = _env("DEFAULT_PROMPT")
    if envp and envp.strip():
        return envp.strip()

    return "object ."


def _choose_output_label(params: dict) -> str:
    """
    Return the MODEL label name (key in CVAT mapping), not the task label.
    Priority: env OUTPUT_LABEL → 'object'
    """
    override = _env("OUTPUT_LABEL")
    if override and override.strip():
        return override.strip()

    return "object"


def _load_image_from_event(context, event) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Return (PIL_Image, params) parsed from:
    - JSON: { image_url: <http(s)>, prompt/labels/... or nested params:{} }
    - multipart/form-data: file field 'image' (+ optional text fields; params as JSON in 'params')
    - image/* or application/octet-stream: raw body is the image
    Also tries to read labels/prompt from headers (x-cvat-labels, x-cvat-prompt).
    """
    params: Dict[str, Any] = {
        # don't prefill 'prompt' here; let incoming values override cleanly
        "box_threshold": float(_env("DEFAULT_BOX_THRESHOLD", "0.20")),
        "text_threshold": float(_env("DEFAULT_TEXT_THRESHOLD", "0.20")),
    }

    # headers
    hdr_labels, hdr_prompt = _labels_from_headers(event)
    if hdr_labels:
        params["labels"] = hdr_labels
    if hdr_prompt:
        params["prompt"] = hdr_prompt

    ct = (event.content_type or "").lower()

    # image/* or octet-stream: raw bytes
    if ct.startswith("image/") or "application/octet-stream" in ct:
        if isinstance(event.body, (bytes, bytearray)) and event.body:
            return _pil_from_bytes(event.body), params
        raise ValueError("Empty binary body for image/* or octet-stream")

    # JSON
    if "application/json" in ct or ct == "":
        try:
            body = event.body
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8") if body else "{}"
            data = body if isinstance(body, dict) else json.loads(body or "{}")

            # merge top-level hints (prompt/labels/mapping/etc.)
            _merge_prompt_hints_into_params(params, data)

            # nested params (dict or JSON string)
            nested = data.get("params")
            if isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except Exception:
                    nested = None
            if isinstance(nested, dict):
                _merge_prompt_hints_into_params(params, nested)

            # thresholds if provided (top and nested)
            for k in ("box_threshold", "text_threshold"):
                if k in data and data[k] not in (None, ""):
                    try:
                        params[k] = float(data[k])
                    except Exception:
                        pass
                if isinstance(nested, dict) and k in nested and nested[k] not in (None, ""):
                    try:
                        params[k] = float(nested[k])
                    except Exception:
                        pass

            # image via URL
            url = data.get("image_url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return _pil_from_bytes(r.content), params

            # image via 'image' field (URL/base64/data-url/bytes)
            im = data.get("image")
            if isinstance(im, str):
                if im.startswith(("http://", "https://")):
                    r = requests.get(im, timeout=30)
                    r.raise_for_status()
                    return _pil_from_bytes(r.content), params
                raw = _bytes_from_possible_base64(im)
                if raw:
                    return _pil_from_bytes(raw), params
            if isinstance(im, (bytes, bytearray)):
                return _pil_from_bytes(im), params

            # some gateways mislabel JSON but send bytes
            if isinstance(event.body, (bytes, bytearray)) and event.body:
                try:
                    return _pil_from_bytes(event.body), params
                except Exception:
                    pass

            raise ValueError("JSON must include 'image_url' or 'image' (URL/base64/data-url/bytes)")
        except Exception as e:
            raise ValueError(f"Failed to parse JSON body: {e}")

    # multipart/form-data
    if "multipart/form-data" in ct:
        fields = getattr(event, "fields", {}) or {}
        files = getattr(event, "files", {}) or {}

        # merge fields (including 'params' JSON if present)
        _merge_prompt_hints_into_params(params, fields)
        if "params" in fields and fields["params"]:
            try:
                nested = json.loads(fields["params"])
                if isinstance(nested, dict):
                    _merge_prompt_hints_into_params(params, nested)
            except Exception:
                pass

        for k in ("box_threshold", "text_threshold"):
            if k in fields and fields[k] not in (None, ""):
                try:
                    params[k] = float(fields[k])
                except Exception:
                    pass

        # file
        if isinstance(files, dict) and "image" in files:
            f = files["image"]
            data = f.get("data") if isinstance(f, dict) else None
            if isinstance(data, (bytes, bytearray)) and data:
                return _pil_from_bytes(data), params

        if isinstance(event.body, (bytes, bytearray)) and event.body:
            return _pil_from_bytes(event.body), params

        raise ValueError("No image provided in multipart/form-data (expected file field 'image')")

    raise ValueError(f"Unsupported Content-Type: {ct}")


def _to_cvat_detections(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    W: int,
    H: int,
    output_label: str,
    min_size: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Convert GroundingDINO outputs to CVAT AA rectangles.
    - Clamp to image bounds (done later)
    - Enforce minimal width/height
    - Use a task-known label name (output_label)
    """
    boxes = boxes.detach().cpu()
    logits = logits.detach().cpu()
    out: List[Dict[str, Any]] = []

    for i in range(min(len(phrases), boxes.shape[0], logits.shape[0])):
        x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]

        # clamp + fix order
        x1 = max(0.0, min(x1, W - 1)); x2 = max(0.0, min(x2, W - 1))
        y1 = max(0.0, min(y1, H - 1)); y2 = max(0.0, min(y2, H - 1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1

        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        score = float(torch.sigmoid(logits[i]).item())
        out.append(
            {
                "label": output_label,  # must exist in the CVAT task
                "type": "rectangle",
                "points": [x1, y1, x2, y2],
                "confidence": score,
            }
        )
    return out


# =========================
# Nuclio entrypoints
# =========================
def init_context(context):
    global _MODEL, _DEVICE

    use_gpu = _env("USE_GPU", "0") == "1"
    _DEVICE = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    torch.set_grad_enabled(False)
    # Optional: cap CPU threads
    # torch.set_num_threads(int(_env("TORCH_NUM_THREADS", "4")))

    cfg_path = _resolve_cfg_path()
    weights_path = _env("WEIGHTS_PATH", "/opt/nuclio/groundingdino.pth")
    weights_path = _ensure_weights(context, weights_path)

    t0 = time.time()
    _MODEL = load_model(cfg_path, weights_path, device=_DEVICE)
    context.logger.info(
        f"GroundingDINO loaded on device={_DEVICE} cfg={cfg_path} in {time.time() - t0:.2f}s"
    )


def handler(context, event):
    try:
        ct = (event.content_type or "")
        body_len = len(event.body) if isinstance(event.body, (bytes, bytearray)) else -1
        context.logger.info(f"invoke ct={ct!r} body_len={body_len}")

        # quick debug of incoming payload (helps verify AA path)
        try:
            hdrs = getattr(event, "headers", {}) or {}
            context.logger.info(f"dbg headers keys={list(hdrs.keys())[:10]}")
        except Exception:
            pass
        try:
            if isinstance(event.body, dict):
                keys = list(event.body.keys())
                preview = {k: event.body[k] for k in keys if k in ("params","mapping","prompt","text","queries","phrases","phrase")}
                context.logger.info(f"dbg body keys={keys} preview={preview}")
        except Exception:
            pass

        # Parse request
        image_pil, params = _load_image_from_event(context, event)
        context.logger.info(f"dbg params={json.dumps(params, ensure_ascii=False)}")

        # Build caption and thresholds
        prompt = _build_caption_from_params(params)
        box_thr = float(params.get("box_threshold", _env("DEFAULT_BOX_THRESHOLD", "0.20")))
        text_thr = float(params.get("text_threshold", _env("DEFAULT_TEXT_THRESHOLD", "0.20")))
        context.logger.info(f"using prompt={prompt!r} box_thr={box_thr} text_thr={text_thr}")

        # Prepare image for model (proper GDINO transforms) and keep sizes
        image_pil, image_t, (W, H), (procW, procH) = _prepare_image_for_model(image_pil)

        # Inference
        boxes, logits, phrases = predict(
            model=_MODEL,
            image=image_t,
            caption=prompt,
            box_threshold=box_thr,
            text_threshold=text_thr,
            device=_DEVICE,
        )

        # 🔧 Robust conversion to ORIGINAL-pixel XYXY (auto-detects format)
        boxes = _boxes_to_xyxy_original(
            boxes, origW=W, origH=H, procW=procW, procH=procH, context=context
        )

        # Choose a task-known label name (so CVAT actually draws)
        output_label = _choose_output_label(params)  # env OUTPUT_LABEL wins
        dets = _to_cvat_detections(boxes, logits, phrases, W, H, output_label)

        # Optional: add a tiny dummy box for plumbing tests (toggle via env)
        if not dets and _env("DEBUG_ADD_DUMMY_BOX", "0") == "1":
            dets.append({"label": output_label, "type": "rectangle", "points": [10, 10, 60, 60], "confidence": 0.01})

        # Detailed logging of all detections
        if not dets:
            context.logger.info("returning 0 dets")
        else:
            context.logger.info(f"returning {len(dets)} dets:")
            for i, d in enumerate(dets):
                pts = d.get("points", [])
                pts_str = (
                    f"[{pts[0]:.1f}, {pts[1]:.1f}, {pts[2]:.1f}, {pts[3]:.1f}]"
                    if isinstance(pts, (list, tuple)) and len(pts) == 4
                    else str(pts)
                )
                conf = d.get("confidence")
                conf_str = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
                context.logger.info(
                    f"  det[{i}] label={d.get('label')} conf={conf_str} points={pts_str} type={d.get('type')}"
                )

        return context.Response(
            body=json.dumps(dets, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
            status_code=200,
        )

    except Exception as e:
        try:
            context.logger.error(f"inference error: {e}")
        except Exception:
            pass
        return context.Response(
            body=json.dumps({"error": str(e)}, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
            status_code=500,
        )
