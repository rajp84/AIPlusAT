import os
import json
import uuid
import threading
import subprocess
from pathlib import Path
import time
from urllib.parse import urlparse
import re

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _stream_output(proc: subprocess.Popen, log_path: Path) -> None:
    try:
        with open(log_path, "a", buffering=1, encoding="utf-8") as f:
            for line in proc.stdout:  # type: ignore[attr-defined]
                f.write(line)
                print(line, end="", flush=True)
    except Exception:
        pass


def _start_background(cmd: list[str], cwd: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_stream_output, args=(proc, log_path), daemon=True).start()
    return proc

def _extract_param_from_event(event, name: str, default=None):
    # Try query dict
    try:
        if hasattr(event, "query") and event.query and name in event.query:
            return event.query.get(name)
    except Exception:
        pass
    # Try headers (common for proxies)
    try:
        hdrs = (event.headers or {})
        # Exact and common variants
        for key in (name, name.upper(), f"x-{name}", f"X-{name}", f"x-{name.replace('_','-')}", f"X-{name.replace('_','-')}"):
            if key in hdrs:
                return hdrs.get(key)
        # Case-insensitive header scan
        lname = name.lower().replace('-', '_')
        for k, v in hdrs.items():
            if isinstance(k, str) and k.lower().replace('-', '_') == lname:
                return v
        # Try common proxy headers that include the URI
        for uri_header in (
            "x-forwarded-uri",
            "X-Forwarded-Uri",
            "x-original-uri",
            "X-Original-Uri",
            "x-request-uri",
            "X-Request-Uri",
            "x-rewrite-url",
            "X-Rewrite-Url",
            ":path",
            "path",
        ):
            if uri_header in hdrs and hdrs.get(uri_header):
                try:
                    parsed = urlparse(hdrs.get(uri_header))
                    if parsed.query:
                        from urllib.parse import parse_qs
                        qs = parse_qs(parsed.query)
                        if name in qs and qs[name]:
                            return qs[name][0]
                except Exception:
                    pass
        # Fallback: parse Referer if present
        if hdrs.get("referer") or hdrs.get("Referer"):
            try:
                parsed = urlparse(hdrs.get("referer") or hdrs.get("Referer"))
                if parsed.query:
                    from urllib.parse import parse_qs
                    qs = parse_qs(parsed.query)
                    if name in qs and qs[name]:
                        return qs[name][0]
            except Exception:
                pass
    except Exception:
        pass
    # Try path query parsing if any
    try:
        parsed = urlparse(getattr(event, "path", ""))
        if parsed.query:
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            if name in qs and qs[name]:
                return qs[name][0]
    except Exception:
        pass
    # Try event.args / event.params / event.parameters
    try:
        for attr in ("args", "params", "parameters"):
            if hasattr(event, attr):
                d = getattr(event, attr)
                if isinstance(d, dict) and name in d:
                    return d.get(name)
    except Exception:
        pass
    # Try full URL if available
    try:
        full_url = getattr(event, "url", None)
        if full_url:
            parsed = urlparse(full_url)
            if parsed.query:
                from urllib.parse import parse_qs
                qs = parse_qs(parsed.query)
                if name in qs and qs[name]:
                    return qs[name][0]
    except Exception:
        pass
    # Try extracting from path segments if looks like a UUID and name == 'id'
    try:
        if name in {"id", "job_id"}:
            raw_path = getattr(event, "path", "") or ""
            for segment in (raw_path or "").split('/')[::-1]:
                if not segment:
                    continue
                if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", segment):
                    return segment
    except Exception:
        pass
    # Try JSON body
    try:
        body = event.body
        if isinstance(body, (bytes, bytearray)):
            body = json.loads(body.decode("utf-8"))
        elif isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict) and name in body:
            return body.get(name)
    except Exception:
        pass
    return default
def _discover_api_root(requests_mod, base_url: str) -> tuple[str, str]:
    """Try to detect whether CVAT API is served under /api or root.
    Returns (base, api_root). api_root is either f"{base}/api" or base.
    """
    base = base_url.rstrip('/')
    candidates = [
        (base + '/api/server/about', base + '/api'),
        (base + '/server/about', base),
    ]
    for probe_url, api_root in candidates:
        try:
            r = requests_mod.get(probe_url, timeout=10)
            print(f"[trainer] Probe {probe_url} -> {r.status_code}", flush=True)
            if r.status_code == 200:
                return base, api_root
        except Exception as e:
            print(f"[trainer] Probe failed {probe_url}: {e}", flush=True)
    # fallback to base/api
    return base, base + '/api'



def _make_response(context, body: dict | str, status_code: int = 200, extra_headers: dict | None = None):
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
        headers = {"Content-Type": "application/json"}
    else:
        headers = {"Content-Type": "text/plain"}
    if extra_headers:
        headers.update(extra_headers)
    return context.Response(body=body, headers=headers, status_code=status_code)


def _update_status(job_id: str):
    with LOCK:
        meta = JOBS.get(job_id)
        if not meta:
            return
        proc: subprocess.Popen = meta["proc"]
        code = proc.poll()
        if code is None:
            meta["status"] = "running"
        elif code == 0:
            meta["status"] = "finished"
        else:
            meta["status"] = "failed"
            meta["exit_code"] = code


def handler(context, event):
    try:
        method = (getattr(event, "method", None) or "").upper()
        path = event.path or "/"
        base_dir = "/opt/nuclio"
        work_dir = os.environ.get("WORK_DIR", f"{base_dir}/workdir")
        jobs_dir = Path(work_dir) / "jobs"

        def _resolve_job_id(endpoint_name: str) -> str | None:
            # Try extracting from the request via robust parser
            jid = _extract_param_from_event(event, "id") or _extract_param_from_event(event, "job_id")
            if jid:
                return str(jid)
            # Fallback: if we have exactly one running job in memory
            with LOCK:
                if len(JOBS) == 1:
                    only = next(iter(JOBS.keys()))
                    return only
            # Fallback: pick the only or latest job log on disk
            try:
                jobs_dir_local = Path(work_dir) / "jobs"
                candidates = list(jobs_dir_local.glob("*.log"))
                if len(candidates) == 1:
                    only_path = candidates[0]
                    only_id = only_path.stem
                    return only_id
                if len(candidates) > 1:
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    latest_id = latest.stem
                    return latest_id
            except Exception:
                pass
            return None
        try:
            # Reduce verbosity: skip routine status/logs noise
            if not (method == "GET" and (path.startswith("/status") or path.startswith("/logs"))):
                hdrs = dict(getattr(event, "headers", {}) or {})
                ct = hdrs.get("content-type") or hdrs.get("Content-Type") or ""
                clen = hdrs.get("content-length") or hdrs.get("Content-Length") or ""
                print(f"[trainer] Incoming request method={method} path={path} ct={ct} len={clen}", flush=True)
        except Exception:
            pass

        # Build CORS headers reflecting caller origin and requested headers
        origin = None
        try:
            origin = (event.headers or {}).get("origin") or (event.headers or {}).get("Origin")
        except Exception:
            origin = None
        allow_origin = origin or "*"
        req_headers = None
        try:
            req_headers = (event.headers or {}).get("access-control-request-headers") or (event.headers or {}).get("Access-Control-Request-Headers")
        except Exception:
            req_headers = None
        cors_headers = {
            "Access-Control-Allow-Origin": allow_origin,
            "Vary": "Origin",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": req_headers or "Content-Type, X-CSRFToken",
            "Access-Control-Max-Age": "86400",
        }
        try:
            # Reduce verbosity: avoid printing CORS line for status/logs endpoints
            if not (method == "GET" and (path.startswith("/status") or path.startswith("/logs"))):
                print(f"[trainer] CORS allow_origin={cors_headers['Access-Control-Allow-Origin']} methods={cors_headers['Access-Control-Allow-Methods']}", flush=True)
        except Exception:
            pass

        # Preflight CORS
        if method == "OPTIONS":
            print("[trainer] Preflight OPTIONS handled", flush=True)
            return _make_response(context, "", 204, cors_headers)

        if method == "POST" and (path == "/" or path.startswith("/train")):
            # Accept either JSON body with params or raw ZIP body with params via query string
            raw_body = event.body or b""
            content_type = ""
            try:
                content_type = (event.headers or {}).get("content-type") or (event.headers or {}).get("Content-Type") or ""
            except Exception:
                content_type = ""
            is_zip_upload = isinstance(raw_body, (bytes, bytearray)) and "application/zip" in content_type
            try:
                print(f"[trainer] Train request zip_upload={is_zip_upload} content_type={content_type}", flush=True)
            except Exception:
                pass

            data = {}
            if not is_zip_upload:
                data = raw_body or {}
                if isinstance(data, (bytes, bytearray)):
                    data = json.loads(data.decode("utf-8"))
                elif isinstance(data, str):
                    data = json.loads(data)

            # Support action multiplexer when called via CVAT lambda proxy
            action = str(data.get("action") or data.get("__action") or "train").lower()
            print(f"[trainer] Action={action}", flush=True)
            if action in {"status", "logs"}:
                job_id = data.get("id") or data.get("job_id")
                if not job_id:
                    return _make_response(context, {"error": "Missing job id"}, 400, cors_headers)
                _update_status(job_id)
                with LOCK:
                    meta = JOBS.get(job_id)
                    if not meta:
                        return _make_response(context, {"error": "Unknown job"}, 404, cors_headers)
                    if action == "status":
                        response_obj = {k: v for k, v in meta.items() if k != "proc"}
                        print(f"[trainer] Status for job={job_id}: {response_obj.get('status')}", flush=True)
                        return _make_response(context, response_obj, 200, cors_headers)
                    # logs
                    log_path = meta.get("log_path")
                try:
                    content = Path(log_path).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
                print(f"[trainer] Logs requested for job={job_id} bytes={len(content)}", flush=True)
                return _make_response(context, {"logs": content}, 200, cors_headers)

            # Supported inputs:
            # - data_yaml: absolute path inside container or uploaded/mounted
            # - cvat_zip: path to a CVAT YOLO 1.1 zip inside container storage
            # - cvat_dir: extracted CVAT folder inside container
            # Optional training args
            def _get_query_param(name: str, default=None):
                try:
                    return event.query.get(name)
                except Exception:
                    return default

            img_size = int((data.get("imgsz") if not is_zip_upload else (_get_query_param("imgsz") or 1280)) or 1280)
            epochs = int((data.get("epochs") if not is_zip_upload else (_get_query_param("epochs") or 100)) or 100)
            batch = int((data.get("batch") if not is_zip_upload else (_get_query_param("batch") or 16)) or 16)
            device = str((data.get("device") if not is_zip_upload else (_get_query_param("device") or "0")) or "0")
            workers = int((data.get("workers") if not is_zip_upload else (_get_query_param("workers") or 8)) or 8)
            model = str((data.get("model") if not is_zip_upload else (_get_query_param("model") or "yolov8s.pt")) or "yolov8s.pt")
            try:
                print(f"[trainer] Params imgsz={img_size} epochs={epochs} batch={batch} device={device} workers={workers} model={model}", flush=True)
            except Exception:
                pass

            # Conversion options
            cvat_zip = data.get("cvat_zip")
            cvat_dir = data.get("cvat_dir")
            out_dataset = data.get("out_dataset", str(Path(work_dir) / "datasets" / str(uuid.uuid4())))
            data_yaml = data.get("data_yaml")
            export_url = data.get("cvat_export_url")

            # If raw ZIP is uploaded, write it to a temp path
            if is_zip_upload:
                try:
                    dl_dir = Path(work_dir) / "uploads"
                    dl_dir.mkdir(parents=True, exist_ok=True)
                    local_zip = dl_dir / f"{uuid.uuid4()}.zip"
                    with open(local_zip, "wb") as f:
                        f.write(raw_body)  # type: ignore[arg-type]
                    cvat_zip = str(local_zip)
                    print(f"[trainer] Stored uploaded zip at {cvat_zip}", flush=True)
                except Exception as e:
                    return _make_response(context, {"error": f"Failed to store uploaded zip: {e}"}, 400, cors_headers)
            # Else if export URL is provided, download it to a temp path and treat as --cvat-zip
            elif export_url:
                try:
                    import requests
                    dl_dir = Path(work_dir) / "downloads"
                    dl_dir.mkdir(parents=True, exist_ok=True)
                    local_zip = dl_dir / f"{uuid.uuid4()}.zip"
                    # If export_url points to localhost from the browser, optionally rewrite to CVAT_BASE_URL if provided
                    try:
                        parsed_eu = urlparse(export_url)
                        if parsed_eu.hostname in {"localhost", "127.0.0.1"}:
                            base_override = os.environ.get("CVAT_BASE_URL")
                            if base_override:
                                base_override = base_override.rstrip("/")
                                export_url = f"{base_override}{parsed_eu.path}{('?' + parsed_eu.query) if parsed_eu.query else ''}"
                                print(f"[trainer] Rewrote export_url to {export_url} via CVAT_BASE_URL", flush=True)
                            else:
                                print("[trainer] export_url host is localhost; will attempt login bases including localhost first", flush=True)
                    except Exception as e:
                        print(f"[trainer] export_url rewrite check failed: {e}", flush=True)
                    # Optional CVAT token for authenticated download
                    auth_token = (
                        data.get("cvat_token")
                        or _get_query_param("cvat_token")
                        or os.environ.get("CVAT_TOKEN")
                    )
                    cvat_user = (
                        data.get("cvat_user")
                        or _get_query_param("cvat_user")
                        or os.environ.get("CVAT_USER")
                    )
                    cvat_pass = (
                        data.get("cvat_pass")
                        or _get_query_param("cvat_pass")
                        or os.environ.get("CVAT_PASS")
                    )
                    host_header = os.environ.get("CVAT_HOST_HEADER")
                    if auth_token:
                        print("[trainer] Using token-based download", flush=True)
                        headers = {"Authorization": f"Token {auth_token}"}
                        if host_header:
                            headers["Host"] = host_header
                        with requests.get(export_url, headers=headers, stream=True, timeout=60) as r:
                            r.raise_for_status()
                            with open(local_zip, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                    elif cvat_user and cvat_pass:
                        # Login with user/pass to obtain a session cookie, then download
                        parsed = urlparse(export_url)
                        url_base_from_export = f"{parsed.scheme}://{parsed.netloc}"
                        # Consider explicit CVAT_BASE_URL env first
                        explicit_base = os.environ.get("CVAT_BASE_URL")
                        bases_to_try = [b for b in [explicit_base, "http://localhost:8080", url_base_from_export, "http://host.docker.internal:8080", "http://cvat:8080", "http://cvat_server:8080"] if b]
                        print(f"[trainer] Base candidates: {bases_to_try}", flush=True)
                        last_err = None
                        with requests.Session() as s:
                            login_payload = {
                                ("email" if "@" in cvat_user else "username"): cvat_user,
                                "password": cvat_pass,
                            }
                            for base_candidate in bases_to_try:
                                base, api_root = _discover_api_root(requests, base_candidate)
                                print(f"[trainer] Using user/pass; base={base} api_root={api_root}", flush=True)
                                # Try POST to {api_root}/auth/login then {base}/auth/login
                                login_urls = [f"{api_root}/auth/login", f"{base}/auth/login"]
                                attempt_log = []
                                for login_url in login_urls:
                                    try:
                                        print(f"[trainer] Attempt login: {login_url}", flush=True)
                                        headers = {"Host": host_header} if host_header else None
                                        lr = s.post(login_url, json=login_payload, headers=headers, timeout=30)
                                        attempt_log.append((login_url, lr.status_code))
                                        lr.raise_for_status()
                                        last_err = None
                                        break
                                    except Exception as le:
                                        last_err = le
                                        try:
                                            lr2 = s.post(login_url, data=login_payload, headers=headers, timeout=30)
                                            attempt_log.append((login_url+"(form)", lr2.status_code))
                                            lr2.raise_for_status()
                                            last_err = None
                                            break
                                        except Exception as le2:
                                            last_err = le2
                                            continue
                                if last_err is None:
                                    # successful login
                                    print(f"[trainer] Login succeeded via {base}", flush=True)
                                    break
                                print(f"[trainer] Login attempts failed for base {base}: {attempt_log}", flush=True)
                            if last_err:
                                raise last_err
                            # Build download URL against the same base to match session cookies
                            dl_url = f"{base}{parsed.path}{('?' + parsed.query) if parsed.query else ''}"
                            print(f"[trainer] Downloading {dl_url}", flush=True)
                            headers = {"Host": host_header} if host_header else None
                            with s.get(dl_url, headers=headers, stream=True, timeout=300) as r:
                                r.raise_for_status()
                                with open(local_zip, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                                        if chunk:
                                            f.write(chunk)
                    else:
                        print("[trainer] Using anonymous download", flush=True)
                        # Anonymous/public download
                        headers = {"Host": host_header} if host_header else None
                        with requests.get(export_url, headers=headers, stream=True, timeout=60) as r:
                            r.raise_for_status()
                            with open(local_zip, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                    cvat_zip = str(local_zip)
                except Exception as e:
                    err_msg = f"Failed to download export: {e}"
                    try:
                        # Include a hint of which method was attempted
                        method_used = (
                            "token" if auth_token else ("userpass" if (cvat_user and cvat_pass) else "anonymous")
                        )
                        err_msg += f" (method={method_used})"
                        if 'base' in locals():
                            err_msg += f" base={base}"
                        if 'api_root' in locals():
                            err_msg += f" api_root={api_root}"
                    except Exception:
                        pass
                    print(f"[trainer] {err_msg}", flush=True)
                    return _make_response(context, {"error": err_msg}, 400, cors_headers)

            job_id = str(uuid.uuid4())
            log_path = jobs_dir / f"{job_id}.log"

            cmd = [
                "python3", "-u",
                str(Path(base_dir) / "train.py"),
            ]

            if cvat_zip or cvat_dir:
                if cvat_zip:
                    cmd += ["--cvat-zip", str(cvat_zip)]
                if cvat_dir:
                    cmd += ["--cvat-dir", str(cvat_dir)]
                cmd += ["--out-dataset", str(out_dataset)]
            elif data_yaml:
                cmd += ["--data", str(data_yaml)]
            else:
                return _make_response(context, {"error": "Provide either data_yaml or cvat-zip/cvat-dir or cvat_export_url"}, 400, cors_headers)

            # Training hyperparams
            cmd += [
                "--imgsz", str(img_size),
                "--epochs", str(epochs),
                "--batch", str(batch),
                "--device", device,
                "--workers", str(workers),
                "--model", model,
            ]
            try:
                print(f"[trainer] Launch cmd: {' '.join(cmd)}", flush=True)
            except Exception:
                pass

            # Export options
            if data.get("build_with_python_trt"):
                cmd += ["--build-with-python-trt"]
            if data.get("build_with_trtexec"):
                cmd += ["--build-with-trtexec"]
            if data.get("trt_half"):
                cmd += ["--trt-half"]
            if data.get("onnx_only"):
                cmd += ["--onnx-only"]

            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(jobs_dir, exist_ok=True)

            # Progress file path for UI/Requests updates
            progress_path = jobs_dir / f"{job_id}.progress.json"
            cmd += ["--progress-file", str(progress_path)]

            # Initialize progress at 10% (dataset prepared/downloaded)
            try:
                progress_path.write_text(json.dumps({"progress": 0.1, "phase": "prepared"}))
            except Exception:
                pass

            # Ensure container tmp uses mounted workdir to avoid filling small container filesystem
            try:
                os.environ.setdefault("TMPDIR", str(Path(work_dir) / "tmp"))
                Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            proc = _start_background(cmd, cwd=base_dir, log_path=log_path)
            with LOCK:
                JOBS[job_id] = {
                    "proc": proc,
                    "status": "running",
                    "log_path": str(log_path),
                    "progress_path": str(progress_path),
                    "cmd": cmd,
                }
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write("[trainer] Job started; launching training...\n")
            except Exception:
                pass
            print(f"[trainer] Started job_id={job_id} log_path={log_path}", flush=True)
            return _make_response(context, {"job_id": job_id}, 200, cors_headers)

        if method == "GET" and path.startswith("/status"):
            job_id = _resolve_job_id("/status")
            if not job_id:
                # Single-job fallback: if only one job exists, use it to avoid 400 during testing
                with LOCK:
                    if len(JOBS) == 1:
                        job_id = next(iter(JOBS.keys()))
                        print(f"[trainer] /status fallback to sole job id={job_id}", flush=True)
                if not job_id:
                    return _make_response(context, {"error": "Missing id"}, 400, cors_headers)
            _update_status(job_id)
            with LOCK:
                meta = JOBS.get(job_id)
            if not meta:
                # Fallback to disk: if log file exists, assume running (unknown exact state)
                candidate = (Path(work_dir) / "jobs" / f"{job_id}.log")
                if candidate.exists():
                    # Try to read progress file if present
                    pfile = Path(work_dir) / "jobs" / f"{job_id}.progress.json"
                    progress = None
                    try:
                        if pfile.exists():
                            progress = json.loads(pfile.read_text()).get("progress")
                    except Exception:
                        pass
                    resp = {"id": job_id, "status": "running", "log_path": str(candidate), "progress": progress}
                    return _make_response(context, resp, 200, cors_headers)
                return _make_response(context, {"error": "Unknown job"}, 404, cors_headers)
            # In-memory: include progress if available
            resp = {k: v for k, v in meta.items() if k != "proc"}
            try:
                ppath = Path(resp.get("progress_path")) if resp.get("progress_path") else None
                if ppath and ppath.exists():
                    resp["progress"] = json.loads(ppath.read_text()).get("progress")
            except Exception:
                pass
            # Intentionally suppress verbose logging for /status calls
            return _make_response(context, resp, 200, cors_headers)

        # Long-polling log stream: returns chunk starting from byte offset, waits up to 25s for new data
        if method == "GET" and path.startswith("/logs/stream"):
            job_id = _resolve_job_id("/logs/stream")
            try:
                offset_val = _extract_param_from_event(event, "offset", 0)
                offset = int(offset_val or 0)
            except Exception:
                offset = 0
            max_wait_s = 25
            if not job_id:
                with LOCK:
                    if len(JOBS) == 1:
                        job_id = next(iter(JOBS.keys()))
                        print(f"[trainer] /logs/stream fallback to sole job id={job_id}", flush=True)
                if not job_id:
                    return _make_response(context, {"error": "Missing id"}, 400, cors_headers)
            log_path = None
            with LOCK:
                meta = JOBS.get(job_id)
                if meta:
                    log_path = meta.get("log_path")
            # Fallback to deterministic path under jobs_dir in case memory state was lost or a different worker handles the request
            if not log_path:
                base_dir = "/opt/nuclio"
                work_dir = os.environ.get("WORK_DIR", f"{base_dir}/workdir")
                jobs_dir = Path(work_dir) / "jobs"
                candidate = jobs_dir / f"{job_id}.log"
                if candidate.exists():
                    log_path = str(candidate)
            if not log_path:
                return _make_response(context, {"error": "Unknown job"}, 404, cors_headers)
            start_time = time.time()
            chunk = ""
            new_offset = offset
            while time.time() - start_time < max_wait_s:
                try:
                    p = Path(log_path)
                    size = p.stat().st_size if p.exists() else 0
                    if size > offset:
                        # read up to 64KB
                        to_read = min(64 * 1024, size - offset)
                        with open(p, "rb") as f:
                            f.seek(offset)
                            data = f.read(to_read)
                        chunk = data.decode("utf-8", errors="ignore")
                        new_offset = offset + len(data)
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            resp = {"chunk": chunk, "offset": new_offset}
            try:
                print(f"[trainer] GET /logs/stream id={job_id} extracted_offset={offset_val} in={offset} out={new_offset} bytes={len(chunk)}", flush=True)
            except Exception:
                pass
            return _make_response(context, resp, 200, cors_headers)

        # Full log fetch (ensure this is after /logs/stream check)
        if method == "GET" and path.startswith("/logs"):
            job_id = _resolve_job_id("/logs")
            if not job_id:
                with LOCK:
                    if len(JOBS) == 1:
                        job_id = next(iter(JOBS.keys()))
                        print(f"[trainer] /logs fallback to sole job id={job_id}", flush=True)
                if not job_id:
                    return _make_response(context, "Missing id", 400, cors_headers)
            log_path = None
            with LOCK:
                meta = JOBS.get(job_id)
                if meta:
                    log_path = meta.get("log_path")
            if not log_path:
                base_dir = "/opt/nuclio"
                work_dir = os.environ.get("WORK_DIR", f"{base_dir}/workdir")
                jobs_dir = Path(work_dir) / "jobs"
                candidate = jobs_dir / f"{job_id}.log"
                if candidate.exists():
                    log_path = str(candidate)
            if not log_path:
                return _make_response(context, "Unknown job", 404, cors_headers)
            try:
                content = Path(log_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""
            try:
                print(f"[trainer] GET /logs id={job_id} bytes={len(content)}", flush=True)
            except Exception:
                pass
            return _make_response(context, content, 200, cors_headers)

        # Discover the latest job id based on job log files
        if method == "GET" and path.startswith("/jobs/latest"):
            try:
                jobs_dir_local = Path(work_dir) / "jobs"
                candidates = list(jobs_dir_local.glob("*.log"))
                if not candidates:
                    return _make_response(context, {"error": "No jobs"}, 404, cors_headers)
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                latest_id = latest.stem
                # Try to return actual status if present
                with LOCK:
                    meta = JOBS.get(latest_id)
                    status = meta.get("status") if meta else "running"
                return _make_response(context, {"job_id": latest_id, "status": status}, 200, cors_headers)
            except Exception as e:
                return _make_response(context, {"error": str(e)}, 500, cors_headers)

        # Health check
        if method == "GET" and path == "/":
            print("[trainer] Health check /", flush=True)
            return _make_response(context, {"status": "ok"}, 200, cors_headers)

        print(f"[trainer] Not found method={method} path={path}", flush=True)
        return _make_response(context, {"error": "Not found"}, 404, cors_headers)
    except Exception as e:
        # Attempt to include CORS even on errors
        try:
            origin = (event.headers or {}).get("origin") or (event.headers or {}).get("Origin")
        except Exception:
            origin = None
        cors_headers = {
            "Access-Control-Allow-Origin": origin or "*",
            "Vary": "Origin",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-CSRFToken",
        }
        try:
            print(f"[trainer] Exception: {e}", flush=True)
        except Exception:
            pass
        return _make_response(context, {"error": str(e)}, 500, cors_headers)


