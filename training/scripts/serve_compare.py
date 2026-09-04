"""Sert la PWA et expose POST /api/analyze (détecteur V5 local)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_village import ROOT as PROJECT_ROOT, analyze_image, load_detector  # noqa: E402

ANALYZE_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
INFERENCE_DIR = ROOT / "training" / "runs" / "inference"


def parse_multipart_image(headers, body: bytes) -> tuple[bytes, str]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("Content-Type multipart/form-data attendu")
    boundary = ""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
    if not boundary:
        raise ValueError("boundary multipart manquante")
    marker = b"--" + boundary.encode("ascii", "ignore")
    for chunk in body.split(marker):
        if not chunk or chunk in (b"--", b"--\r\n"):
            continue
        header_blob, _, file_body = chunk.partition(b"\r\n\r\n")
        header_text = header_blob.decode("utf-8", "replace").lower()
        if "name=\"image\"" not in header_text and "name=image" not in header_text:
            continue
        filename = "village.png"
        for line in header_text.split("\r\n"):
            if "filename=" in line:
                raw = line.split("filename=", 1)[1].strip().strip("\"'")
                if raw:
                    filename = Path(unquote(raw)).name
        data = file_body.rstrip(b"\r\n")
        if data.endswith(b"--"):
            data = data[:-2].rstrip(b"\r\n")
        if not data:
            raise ValueError("image vide")
        return data, filename
    raise ValueError("champ image manquant")


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "engine": "building-detector-v5s-infer800",
                    "imgsz": 800,
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/analyze":
            self._json(404, {"error": "route inconnue"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._json(400, {"error": "image absente ou trop lourde (max 20 Mo)"})
            return
        body = self.rfile.read(length)
        tmp_path = None
        try:
            data, filename = parse_multipart_image(self.headers, body)
            suffix = Path(filename).suffix or ".png"
            INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=str(INFERENCE_DIR)
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(data)
            with ANALYZE_LOCK:
                payload = analyze_image(tmp_path, output_dir=None)
            payload.pop("source", None)
            self._json(200, payload)
        except Exception as exc:  # noqa: BLE001 — renvoyer l'erreur à l'UI locale
            self._json(500, {"error": str(exc)})
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))
        sys.stderr.flush()


def main() -> None:
    port = int(os.environ.get("CLASHCOMPARE_PORT", "8765"))
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    print("Chargement du détecteur V5…", flush=True)
    load_detector(PROJECT_ROOT / "models" / "building-detector.pt")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ClashCompare: http://127.0.0.1:{port}/", flush=True)
    print("POST /api/analyze  GET /api/health", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
