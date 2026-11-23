"""CPU regression tests for local-only ECS fault injection and validation."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

TOOLS = Path(__file__).resolve().parents[1] / "scripts/tools"


def load(name):
    """Load standalone tooling without importing training/evaluation extras."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proxy_module = load("mlflow_fault_proxy")
validation = load("validate_mlflow_ecs")


@pytest.fixture
def backend():
    """Run a disposable HTTP target; no real tracking service is contacted."""
    received = []
    status = [200]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(status[0])
            self.end_headers()
            self.wfile.write(b"{}")

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"healthy")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received, status
    server.shutdown()
    server.server_close()
    thread.join()


def payload(step, training=True):
    """Match the MLflow log-batch wire format."""
    return {
        "run_id": "test-run",
        "metrics": [
            {"key": "global_step", "step": step, "value": step, "timestamp": 1},
            {
                "key": "train/loss" if training else "system/cpu",
                "step": step,
                "value": 1,
                "timestamp": 1,
            },
        ],
    }


@pytest.mark.parametrize("wire_step", [1, "1"])
def test_midflight_cut_is_local_and_requires_successful_training(backend, wire_step):
    """Initialization, system metrics and failed requests must not arm the cut."""
    upstream, received, status = backend
    with proxy_module.TrackingFaultProxy(upstream) as proxy:
        url = proxy.uri + "/api/2.0/mlflow/runs/log-batch"
        for data in (payload(0), payload(1, training=False)):
            requests.post(url, json=data, timeout=2).raise_for_status()
            assert not proxy.cut.is_set()
        status[0] = 500
        assert requests.post(url, json=payload(1), timeout=2).status_code == 500
        assert not proxy.cut.is_set()
        status[0] = 200
        data = payload(1)
        for metric in data["metrics"]:
            metric["step"] = wire_step
        requests.post(url, json=data, timeout=2).raise_for_status()
        assert proxy.cut.wait(2)
        assert proxy.run_id == "test-run"
        before = len(received)
        with pytest.raises(requests.ConnectionError):
            requests.post(url, json=payload(2), timeout=2)
        assert len(received) == before
        assert proxy.rejected_requests == 1
        assert requests.get(upstream + "/health", timeout=2).text == "healthy"


@pytest.mark.parametrize(
    "uri",
    ["file:///tmp/store", "http://user:secret@localhost", "http://localhost?token=x"],
)
def test_proxy_rejects_unsafe_target(uri):
    """No credential-bearing URL is accepted into subprocess/logged config."""
    with pytest.raises(ValueError):
        proxy_module.TrackingFaultProxy(uri)


def test_gpu_requires_distinct_compute_and_memory_series():
    """A memory utilization series alone cannot falsely satisfy GPU validation."""
    with pytest.raises(AssertionError):
        validation.gpu_metrics({"system/gpu_0_memory_utilization_percentage": 1})
    good = {
        "system/gpu_0_utilization_percentage": 0,
        "system/gpu_0_memory_usage_megabytes": 12,
    }
    assert validation.gpu_metrics(good) == good
    with pytest.raises(AssertionError):
        validation.gpu_metrics(
            {
                "system/gpu_0_utilization_percentage": 1,
                "system/gpu_1_memory_usage_megabytes": 12,
            }
        )


def test_timeout_kills_scoped_child_processes(tmp_path):
    """A wrapper timeout also stops its training child, not unrelated processes."""
    marker = tmp_path / "survived"
    child = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).touch()"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    with (
        (tmp_path / "job.log").open("w") as log,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        validation.run_process(
            [sys.executable, "-c", parent],
            cwd=tmp_path,
            env=os.environ.copy(),
            log=log,
            timeout=0.3,
        )
    time.sleep(1.1)
    assert not marker.exists()


def test_callback_warn_continues_after_midflight_http_disconnect(backend, tmp_path):
    """Exercise real Callback degradation through the real loopback HTTP proxy."""
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "_fault_callback",
        TOOLS.parents[1] / "olmoearth_pretrain/train/callbacks/mlflow.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    upstream, received, _ = backend
    with proxy_module.TrackingFaultProxy(upstream) as proxy:

        class Client:
            def log_batch(self, run_id, metrics, **kwargs):
                data = {
                    "run_id": run_id,
                    "metrics": [
                        {
                            "key": m.key,
                            "value": m.value,
                            "timestamp": m.timestamp,
                            "step": m.step,
                        }
                        for m in metrics
                    ],
                }
                requests.post(
                    proxy.uri + "/api/2.0/mlflow/runs/log-batch", json=data, timeout=2
                ).raise_for_status()

        callback = module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, run_id="test-run", log_interval=1
        )
        callback.trainer = SimpleNamespace(global_step=1)
        callback._client = Client()
        callback.log_metrics(1, {"train/loss": 3.0})
        assert proxy.cut.wait(2)
        callback.log_metrics(2, {"train/loss": 2.0})
        callback.trainer.global_step = 3
        callback.log_metrics(3, {"train/loss": 1.0})
        assert callback._disabled
        assert callback.state_dict()["run_id"] == "test-run"
        callback.post_train()
        callback.close()
        assert len(received) == 1
        assert requests.get(upstream, timeout=2).status_code == 200


def test_no_cuda_does_not_create_output_directory(tmp_path, monkeypatch):
    """A failed environment preflight leaves the requested path reusable."""
    output = tmp_path / "not-created"
    monkeypatch.setenv("MLFLOW_VALIDATION_DIR", str(output))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setattr(validation.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        validation.main()
    assert not output.exists()
