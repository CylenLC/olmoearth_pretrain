"""Validate Nano tracking on a CUDA host against a separately deployed server."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
from mlflow import MlflowClient


def main() -> None:
    """Run isolated training, resume and unavailable-server checks."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    root = Path(__file__).resolve().parents[2]
    output = Path(
        os.environ.get("MLFLOW_VALIDATION_DIR", "/tmp/olmoearth-mlflow-validation")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    client = MlflowClient(tracking_uri=uri)
    client.search_experiments(max_results=1)
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
        env.update(
            MLFLOW_HTTP_REQUEST_TIMEOUT="2",
            MLFLOW_HTTP_REQUEST_MAX_RETRIES="0",
            MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL="0.1",
            MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING="1",
        )
        with (output / f"{name}-{steps}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def identity(name: str, step: int) -> str | None:
        state = torch.load(
            output / name / "checkpoints" / f"step{step}" / "train/rank0.pt",
            map_location="cpu",
            weights_only=False,
        )
        return state["callbacks"]["mlflow"]["run_id"]

    run("online", 2, uri, "raise")
    first = identity("online", 2)
    if not first:
        raise AssertionError("No Run ID in checkpoint")
    run("online", 3, uri, "raise")
    if identity("online", 3) != first:
        raise AssertionError("Run ID changed after resume")
    tracked = client.get_run(first)
    gpu = {
        k: v
        for k, v in tracked.data.metrics.items()
        if k.startswith("system/") and "gpu" in k.lower()
    }
    if not any("utilization" in k for k in gpu) or not any("memory" in k for k in gpu):
        raise AssertionError(f"Missing GPU utilization/memory metrics: {gpu}")
    if not any(m.step == 3 for m in client.get_metric_history(first, "global_step")):
        raise AssertionError("Missing resumed step 3")
    # Reserve a closed local port; never stop or modify the real tracking server.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        unavailable = f"http://127.0.0.1:{reserved.getsockname()[1]}"
        run("offline", 2, unavailable, "warn")
    if identity("offline", 2) is not None:
        raise AssertionError("Unexpected Run ID for unreachable server")
    summary = {
        "run_id": first,
        "resumed_step": 3,
        "gpu_metrics": gpu,
        "unavailable_server_training": "completed two steps",
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
