"""Loopback-only fault injection for the ECS validation command.

Forwards to one fixed server until step-one training metrics succeed, then
closes subsequent client connections. Never stops or reconfigures the server.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}


def validate_tracking_uri(upstream: str) -> None:
    """Reject unsupported URLs before they can enter subprocess logs."""
    parsed = urlsplit(upstream)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "Use MLflow authentication environment variables, not URL credentials"
        )


class TrackingFaultProxy:
    """Forward HTTP tracking traffic and cut only this validation client's route."""

    def __init__(self, upstream: str, verify: str | bool = True):
        """Bind a fresh loopback port; prohibit credentials in the target URL."""
        validate_tracking_uri(upstream)
        self.upstream = upstream.rstrip("/")
        self.cut = threading.Event()
        self.run_id = None
        self.rejected_requests = 0
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                # HTTP paths and headers can contain credentials.
                pass

            def do_GET(self):
                self.forward()

            def do_POST(self):
                self.forward()

            def do_PUT(self):
                self.forward()

            def forward(self):
                # Serialize the cut boundary, including system-monitor requests.
                with owner._lock:
                    if owner.cut.is_set():
                        owner.rejected_requests += 1
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                        return
                    if not self.path.startswith("/") or self.path.startswith("//"):
                        self.send_error(400)
                        return
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    headers = {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower() not in _HOP_HEADERS
                    }
                    try:
                        response = requests.request(
                            self.command,
                            owner.upstream + self.path,
                            headers=headers,
                            data=body,
                            timeout=5,
                            allow_redirects=False,
                            verify=verify,
                        )
                    except requests.RequestException:
                        self.send_error(502, "Upstream request failed")
                        return
                    with response:
                        content = response.content
                        self.send_response(response.status_code)
                        for key, value in response.headers.items():
                            if key.lower() not in _HOP_HEADERS:
                                self.send_header(key, value)
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                        if 200 <= response.status_code < 300 and self.path.split(
                            "?", 1
                        )[0].endswith("/runs/log-batch"):
                            payload = json.loads(body)
                            metrics = payload.get("metrics", [])
                            # Checkpoints also emit global_step. Require actual
                            # training metrics, not initialization/system data.
                            if any(
                                m["key"].startswith("train/") for m in metrics
                            ) and any(
                                m["key"] == "global_step" and int(m["step"]) >= 1
                                for m in metrics
                            ):
                                owner.run_id = payload["run_id"]
                                owner.cut.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.uri = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        """Start forwarding for the validation subprocess."""
        self.thread.start()
        return self

    def __exit__(self, *args):
        """Release only the local proxy's resources."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
