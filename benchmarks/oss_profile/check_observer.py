"""Exercise the actual prefetch alias, boto3 send and StreamingBody on loopback."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

import boto3
from botocore.config import Config
from upath import UPath

import observer
from olmoearth_pretrain.data import dataloader


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.calls += 1
        if self.server.calls == 1:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "3")
        self.end_headers()
        self.wfile.write(b"abc")

    def log_message(self, *args):
        pass


def check():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        os.environ["OSS_PROFILE_OUTPUT"] = directory
        observer.open_log("fixture")
        if "--without-observer" not in sys.argv:
            observer.install_get()
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            server.calls = 0
            thread.start()
            client = boto3.client(
                "s3", endpoint_url=f"http://127.0.0.1:{server.server_port}",
                aws_access_key_id="fixture", aws_secret_access_key="fixture",
                config=Config(retries={"total_max_attempts": 2, "mode": "standard"}),
            )
            prefetcher = dataloader._OSSGetPrefetcher(2)
            try:
                with patch("olmoearth_pretrain.data.dataset._get_oss_client", return_value=client):
                    path = UPath("s3://fixture/sample_1.h5", client_kwargs={
                        "endpoint_url": "https://fixture.aliyuncs.com",
                    })
                    assert prefetcher.submit(path).result(timeout=5) == b"abc"
            finally:
                prefetcher.close()
                client.close()
                server.shutdown()
                thread.join()
        with next(Path(directory).glob("*.jsonl")).open() as stream:
            rows = [json.loads(line) for line in stream]
        observer._stream.close()
        starts = [row for row in rows if row["event"] == "get_start"]
        ends = [row for row in rows if row["event"] == "get_end"]
        assert len(starts) == len(ends) == 1, "worker pool GET must emit start and end"
        end = ends[0]
        assert end["attempts"] == 2 and end["sdk_retries"] == 1, end
        assert end["attempt_status"] == [503, 200] and end["status"] == 200, end
        assert end["expected_bytes"] == end["body_bytes"] == end["bytes"] == 3, end
        assert end["start_ns"] <= end["request_ns"] <= end["headers_ns"] <= (
            end["body_start_ns"]
        ) <= end["body_end_ns"] <= end["end_ns"], end
        print("GET observer verified: one full object, two attempts, three body bytes")


if __name__ == "__main__":
    check()
