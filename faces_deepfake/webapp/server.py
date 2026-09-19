"""Small local web UI for the DeepFake Detector models.

Run from the repository root:
    python faces_deepfake/webapp/server.py
Then open http://127.0.0.1:8000.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import tempfile
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

APP_DIR = Path(__file__).resolve().parent
FACES_DIR = APP_DIR.parent
USER_DIR = FACES_DIR / "user"
REPORT_PATH = USER_DIR / "model_parameters_report.md"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for directory in (USER_DIR, FACES_DIR / "models", FACES_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


class DetectorHandler(BaseHTTPRequestHandler):
    server_version = "DeepFakeDetector/1.0"

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/model-report":
            self.send_json({"report": REPORT_PATH.read_text(encoding="utf-8")})
            return
        if route == "/api/health":
            self.send_json({"status": "ok"})
            return
        self.serve_static(route)

    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            result = self.analyze_upload()
            self.send_json(result)
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # Do not expose a traceback to the browser.
            print(f"[ERROR] Analysis failed: {error}", file=sys.stderr)
            self.send_json(
                {"error": "Analysis could not be completed. Confirm the model files and Python dependencies are installed."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def analyze_upload(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if not content_length:
            raise ValueError("Choose an image first.")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError("Image is too large. The limit is 12 MB.")

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected an image upload.")
        raw_body = self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + raw_body
        )
        uploaded = next((part for part in message.iter_attachments() if part.get_param("name", header="content-disposition") == "image"), None)
        filename = uploaded.get_filename() if uploaded else None
        if uploaded is None or not filename:
            raise ValueError("Choose an image first.")

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Use JPG, PNG, WEBP, or BMP images.")
        payload = uploaded.get_payload(decode=True) or b""
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("Image is too large. The limit is 12 MB.")

        # PIL both validates the uploaded bytes and avoids passing arbitrary files to Torch.
        from PIL import Image
        from io import BytesIO

        try:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
        except Exception as error:
            raise ValueError("The uploaded file is not a valid image.") from error

        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(payload)
                temp_name = temp_file.name
            from use_weighted_fusion import predict_image
            from explainability import generate_all_explanations, generate_patch_heatmap

            prediction = predict_image(temp_name)
            prediction["filename"] = Path(filename).name
            prediction.pop("image_path", None)

            # Generate Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, and LIME heatmaps
            try:
                target_verdict = prediction.get("prediction", "FAKE")
                xai_results = generate_all_explanations(temp_name, target_class=target_verdict)
                prediction["heatmaps"] = xai_results["overlays"]
                prediction["raw_heatmaps"] = xai_results["raw_heatmaps"]
                prediction["descriptions"] = xai_results["descriptions"]
                prediction["target_explanation"] = xai_results["target_class"]
                prediction["transformed_image"] = xai_results.get("transformed_image")
            except Exception as xai_err:
                print(f"[WARN] Explainability generation failed: {xai_err}", file=sys.stderr)
                prediction["heatmaps"] = {}
                prediction["raw_heatmaps"] = {}
                prediction["descriptions"] = {}
                prediction["target_explanation"] = None
                prediction["transformed_image"] = None

            # Generate Simple Patch Heatmap for Local Model
            try:
                patch_probs = prediction.get("local_model", {}).get("patch_probs", [])
                if patch_probs:
                    patch_xai = generate_patch_heatmap(
                        patch_probs,
                        image_input=temp_name
                    )
                    prediction["local_model"]["patch_heatmap_overlay"] = patch_xai["overlay"]
                    prediction["local_model"]["patch_raw_heatmap"] = patch_xai["raw_heatmap"]
                    prediction["local_model"]["patch_grid"] = patch_xai["patch_grid"]
            except Exception as patch_err:
                print(f"[WARN] Patch heatmap generation failed: {patch_err}", file=sys.stderr)

            return prediction
        finally:
            if temp_name and os.path.exists(temp_name):
                os.remove(temp_name)

    def serve_static(self, route: str):
        requested = "index.html" if route in ("", "/") else route.lstrip("/")
        path = (APP_DIR / requested).resolve()
        if APP_DIR not in path.parents and path != APP_DIR:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"[WEB] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DetectorHandler)
    print("DeepFake Detector web app: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
