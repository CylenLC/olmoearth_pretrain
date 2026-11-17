"""Validate Nano tracking on a CUDA host against a separately deployed server."""

import json
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

import torch
from mlflow import MlflowClient
from mlflow_fault_proxy import TrackingFaultProxy, validate_tracking_uri


def gpu_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Require compute utilization and memory usage from the same GPU."""
    gpu = {k: v for k, v in metrics.items() if k.startswith("system/") and "gpu_" in k}
    for key in gpu:
        match = re.fullmatch(r"(.*/gpu_\d+)_utilization_percentage", key)
        if match and match[1] + "_memory_usage_megabytes" in gpu:
            return gpu
    raise AssertionError(
        "Missing separate GPU compute utilization and memory usage metrics"
    )


def run_process(command, *, cwd, env, log, timeout):
    """Bound execution and kill only this validation job's process group."""
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def main() -> None:
    """Run isolated training, resume and unavailable-server checks."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    validate_tracking_uri(uri)
    root = Path(__file__).resolve().parents[2]
    output = Path(
        os.environ.get("MLFLOW_VALIDATION_DIR", "/tmp/olmoearth-mlflow-validation")
    ).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "5"
    timeout = int(os.environ.get("MLFLOW_VALIDATION_TIMEOUT", "1800"))
    if timeout <= 0:
        raise ValueError("MLFLOW_VALIDATION_TIMEOUT must be positive")
    client = MlflowClient(tracking_uri=uri)
    client.search_experiments(max_results=1)
    output.mkdir(parents=True, exist_ok=False)
    experiment = "ecs-validation-" + output.name

    def run(name: str, steps: int, tracking_uri: str, policy: str) -> None:
        command = [
            sys.executable,
            str(root / "scripts/tools/mlflow_nano_smoke.py"),
            "--work-dir",
            str(output / name),
            "--mlflow",
            "--steps",
            str(steps),
            "--tracking-uri",
            tracking_uri,
            f"--trainer.callbacks.mlflow.experiment_name={experiment}",
            f"--trainer.callbacks.mlflow.failure_policy={policy}",
            "--trainer.callbacks.mlflow.system_metrics_interval=0.1",
        ]
        env = os.environ.copy()
        # Keep the fault route on this host even in proxy-configured environments.
        bypass = ",".join(
            filter(
                None, [env.get("NO_PROXY"), env.get("no_proxy"), "127.0.0.1,localhost"]
            )
        )
        env["NO_PROXY"] = env["no_proxy"] = bypass
        env.update(
            MLFLOW_HTTP_REQUEST_TIMEOUT="2",
            MLFLOW_HTTP_REQUEST_MAX_RETRIES="0",
            MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL="0.1",
            MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING="1",
        )
        with (output / f"{name}-{steps}.log").open("w") as log:
            run_process(
                command,
                cwd=root,
                env=env,
                log=log,
                timeout=timeout,
            )

    def identity(name: str, step: int) -> str | None:
        state = torch.load(
            output / name / "checkpoints" / f"step{step}" / "train/rank0.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state["global_step"] != step:
            raise AssertionError("Checkpoint global_step does not match target")
        return state["callbacks"]["mlflow"]["run_id"]

    run("online", 2, uri, "raise")
    first = identity("online", 2)
    if not first:
        raise AssertionError("No Run ID in checkpoint")
    run("online", 3, uri, "raise")
    if identity("online", 3) != first:
        raise AssertionError("Run ID changed after resume")
    tracked = client.get_run(first)
    gpu = gpu_metrics(tracked.data.metrics)
    if not any(m.step == 3 for m in client.get_metric_history(first, "global_step")):
        raise AssertionError("Missing resumed step 3")
    # Reserve a closed local port; never stop or modify the real tracking server.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        unavailable = f"http://127.0.0.1:{reserved.getsockname()[1]}"
        run("offline", 2, unavailable, "warn")
    if identity("offline", 2) is not None:
        raise AssertionError("Unexpected Run ID for unreachable server")
    # Only this subprocess uses the loopback proxy. The real server stays up.
    with TrackingFaultProxy(
        uri, verify=os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH") or True
    ) as proxy:
        run("midflight", 3, proxy.uri, "warn")
        midflight_id = identity("midflight", 3)
        if not proxy.cut.is_set() or proxy.rejected_requests == 0:
            raise AssertionError("Mid-training disconnect was not exercised")
        if not midflight_id or midflight_id != proxy.run_id:
            raise AssertionError("Checkpoint lost the run identity after disconnect")
        log_text = (output / "midflight-3.log").read_text()
        if "MLflow reporting disabled after" not in log_text:
            raise AssertionError("No evidence that warn-mode downgrade occurred")
        history = client.get_metric_history(midflight_id, "global_step")
        if not any(m.step == 1 for m in history) or any(m.step > 1 for m in history):
            raise AssertionError("Unexpected server metric history across disconnect")
        # A direct read proves the actual server is still reachable.
        client.get_run(midflight_id)
        rejected = proxy.rejected_requests
    summary = {
        "run_id": first,
        "midflight_run_id": midflight_id,
        "midflight_checkpoint_step": 3,
        "proxy_rejected_requests": rejected,
        "resumed_step": 3,
        "gpu_metrics": gpu,
        "unavailable_server_training": "completed two steps",
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
