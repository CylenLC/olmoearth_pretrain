"""MLflow callback contract tests using the real client and a local tracking store.

These tests deliberately avoid the repository-wide geospatial fixtures. The
callback is loaded directly so optional downstream evaluation packages are not
needed to verify this lightweight integration.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

CALLBACK_PATH = (
    Path(__file__).resolve().parents[1] / "olmoearth_pretrain/train/callbacks/mlflow.py"
)


def load_callback_module(name: str = "_olmoearth_mlflow_test") -> Any:
    """Load only the callback, without the optional evaluation package tree."""
    spec = importlib.util.spec_from_file_location(name, CALLBACK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module() -> Any:
    """Load a fresh callback module for independent runtime state."""
    return load_callback_module()


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Use an isolated real MLflow FileStore, never the user's server."""
    import mlflow
    from mlflow import MlflowClient

    old_uri = mlflow.get_tracking_uri()
    while mlflow.active_run() is not None:
        mlflow.end_run()
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    yield SimpleNamespace(mlflow=mlflow, client=client, uri=uri)
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(old_uri)


def attach(callback: Any, tmp_path: Path, step: int = 0) -> Any:
    """Attach the minimal Trainer contract used by tracking hooks."""
    callback.trainer = SimpleNamespace(
        global_step=step,
        save_folder=str(tmp_path / "checkpoints"),
        data_loader=SimpleNamespace(dataset=[None] * 12),
        checkpoint_loaded=False,
        callbacks={"mlflow": callback},
    )
    return callback


def finish(callback: Any) -> None:
    """Mirror Trainer's successful post_train/close lifecycle."""
    callback.post_train()
    callback.close()


def artifact_text(client: Any, run_id: str) -> str:
    """Read only tiny callback artifacts from the local store."""
    chunks: list[str] = []

    def visit(path: str | None = None) -> None:
        for artifact in client.list_artifacts(run_id, path):
            if artifact.is_dir:
                visit(artifact.path)
            else:
                local = Path(client.download_artifacts(run_id, artifact.path))
                assert local.stat().st_size < 1_000_000
                chunks.append(local.read_text())

    visit()
    return "\n".join(chunks)


@pytest.mark.parametrize("enabled,rank", [(False, 0), (True, 1), (True, 7)])
def test_disabled_and_nonzero_ranks_do_not_import_client(
    enabled: bool, rank: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled/rank-filtered tracking performs no optional client work."""
    original_import = builtins.__import__
    attempted: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow" or name.startswith("mlflow."):
            attempted.append(name)
            raise AssertionError("MLflow must remain lazy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = load_callback_module("_olmoearth_mlflow_disabled_test")
    monkeypatch.setattr(module, "get_rank", lambda: rank)
    callback = attach(module.OlmoEarthMLflowCallback(enabled=enabled), tmp_path)
    callback.pre_train()
    callback.log_metrics(10, {"train/loss": 0.5})
    callback.post_checkpoint_saved("oss://bucket/run/step10")
    callback.log_evaluation("task", {"accuracy": 0.8}, step=10)
    finish(callback)
    assert attempted == []


def test_config_metrics_and_global_step(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """Existing numeric metrics retain names, values and Trainer step."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            experiment_name="unit-comparison",
            run_name="nano-one",
            log_interval=2,
            system_metrics=False,
            config={"init_seed": 12536, "train_module": {"optim": {"lr": 0.002}}},
        ),
        tmp_path,
    )
    callback.pre_train()
    callback.log_metrics(1, {"train/loss": 4.0})
    metrics = {"train/loss": 2.0, "optim/LR": 0.002, "optim/grad_norm": 0.4}
    original_metrics = dict(metrics)
    callback.log_metrics(2, metrics)
    finish(callback)
    assert metrics == original_metrics
    run_id = callback.state_dict()["run_id"]
    run = tracking.client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["init_seed"] == "12536"
    assert run.data.tags["mlflow.runName"] == "nano-one"
    history = tracking.client.get_metric_history(run_id, "train/loss")
    assert [(metric.step, metric.value) for metric in history] == [(1, 4.0), (2, 2.0)]
    global_step = tracking.client.get_metric_history(run_id, "global_step")
    assert (global_step[-1].step, global_step[-1].value) == (2, 2.0)
    assert run.data.metrics["optim/grad_norm"] == 0.4


def test_environment_uri_precedes_config_and_experiments_compare(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """The documented environment override selects the server for each run."""
    ids = []
    for name in ("first", "second"):
        callback = attach(
            module.OlmoEarthMLflowCallback(
                enabled=True,
                tracking_uri="http://127.0.0.1:1",
                experiment_name="comparison",
                run_name=name,
                system_metrics=False,
            ),
            tmp_path,
        )
        callback.pre_train()
        finish(callback)
        ids.append(callback.state_dict()["run_id"])
    assert ids[0] != ids[1]
    runs = [tracking.client.get_run(run_id) for run_id in ids]
    assert runs[0].info.experiment_id == runs[1].info.experiment_id


def test_dataset_metadata_without_reading_source(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """MLflow DatasetInput describes an unreachable OSS URI without fetching it."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            system_metrics=False,
            dataset_name="earth-tiles",
            dataset_version="2026-11",
            dataset_uri="oss://deliberately-inaccessible-bucket/pretrain/v2",
            dataset_num_samples=123456,
            dataset_preprocessing={"normalize": True, "patch_size": 8},
        ),
        tmp_path,
    )
    callback.pre_train()
    finish(callback)
    run_id = callback.state_dict()["run_id"]
    run = tracking.client.get_run(run_id)
    assert len(run.inputs.dataset_inputs) == 1
    dataset = run.inputs.dataset_inputs[0].dataset
    assert dataset.name == "earth-tiles"
    assert "oss://deliberately-inaccessible-bucket/pretrain/v2" in dataset.source
    text = artifact_text(tracking.client, run_id)
    for expected in ("2026-11", "123456", "normalize", "patch_size"):
        assert expected in text


def test_checkpoint_uri_and_resume_preserve_run(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """Checkpoint state restores one run and async completion keeps saved step."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, log_interval=1
        ),
        tmp_path,
        step=10,
    )
    callback.pre_train()
    callback.log_metrics(10, {"train/loss": 0.75})
    state = callback.state_dict()
    run_id = state["run_id"]
    callback.trainer.global_step = 13  # async save completed several steps later
    callback.post_checkpoint_saved("oss://bucket/experiment/step10")
    finish(callback)
    text = artifact_text(tracking.client, run_id)
    assert "oss://bucket/experiment/step10" in text
    assert "0.75" in text
    checkpoint_records = []
    for artifact in tracking.client.list_artifacts(run_id, "checkpoints"):
        if artifact.path.endswith(".json"):
            checkpoint_records.append(
                json.loads(
                    Path(
                        tracking.client.download_artifacts(run_id, artifact.path)
                    ).read_text()
                )
            )
    assert any(record["step"] == 10 for record in checkpoint_records)
    resumed = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, log_interval=1
        ),
        tmp_path,
        step=10,
    )
    resumed.load_state_dict(state)
    resumed.pre_train()
    assert resumed.state_dict()["run_id"] == run_id
    resumed.log_metrics(11, {"train/loss": 0.5})
    finish(resumed)
    history = tracking.client.get_metric_history(run_id, "train/loss")
    assert [(metric.step, metric.value) for metric in history] == [
        (10, 0.75),
        (11, 0.5),
    ]


def test_secret_redaction_and_environment_allowlist(
    module: Any, tracking: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials in configs, signed URIs and ambient env never reach artifacts."""
    monkeypatch.setenv("UNRELATED_PASSWORD", "ambient-secret-value")
    config = {
        "init_seed": 12,
        "access_key": "access-key-value",
        "AccessKeyId": "access-key-id-value",
        "nested": {"token": "token-secret-value", "password": "password-secret-value"},
        "source": "https://user:basic-secret-value@example.com/data?signature=url-secret-value",
    }
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, config=config
        ),
        tmp_path,
    )
    callback.pre_train()
    callback.log_evaluation(
        "downstream",
        {"accuracy": 0.8},
        config={"api_key": "evaluation-key-value"},
        results={"output": "oss://bucket/eval?X-Amz-Signature=artifact-secret-value"},
        step=5,
    )
    finish(callback)
    run_id = callback.state_dict()["run_id"]
    run = tracking.client.get_run(run_id)
    text = artifact_text(tracking.client, run_id) + json.dumps(run.to_dictionary())
    for secret in (
        "ambient-secret-value",
        "access-key-value",
        "access-key-id-value",
        "token-secret-value",
        "password-secret-value",
        "basic-secret-value",
        "url-secret-value",
        "evaluation-key-value",
        "artifact-secret-value",
    ):
        assert secret not in text
    assert config["nested"]["token"] == "token-secret-value"


def test_evaluation_attaches_without_finishing_training_run(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """A separate evaluation process can attach to a still-running training run."""
    experiment_id = tracking.client.create_experiment("training")
    run_id = tracking.client.create_run(experiment_id).info.run_id
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            run_id=run_id,
            evaluation_only=True,
            system_metrics=False,
        ),
        tmp_path,
    )
    callback.pre_train()
    callback.log_evaluation(
        "classification", {"accuracy": 0.9}, config={"split": "test"}, step=100
    )
    finish(callback)
    run = tracking.client.get_run(run_id)
    assert run.info.status == "RUNNING"
    assert "accuracy" in artifact_text(tracking.client, run_id)
    assert any(
        "accuracy" in name and value == 0.9 for name, value in run.data.metrics.items()
    )


def test_system_metrics_use_mlflow_monitor(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """The installed MLflow monitor reports real host metrics into the run."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            system_metrics=True,
            system_metrics_interval=0.1,
        ),
        tmp_path,
    )
    callback.pre_train()
    run_id = callback.state_dict()["run_id"]
    deadline = time.monotonic() + 5
    metrics: dict[str, float] = {}
    while time.monotonic() < deadline:
        metrics = tracking.client.get_run(run_id).data.metrics
        if any("cpu_utilization" in key for key in metrics) and any(
            "memory" in key for key in metrics
        ):
            break
        time.sleep(0.1)
    finish(callback)
    assert any("cpu_utilization" in key for key in metrics)
    assert any("memory" in key for key in metrics)


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_missing_optional_client_obeys_failure_policy(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """Training may continue without MLflow or opt into an explicit error."""
    original_import = builtins.__import__

    def unavailable(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("MLflow is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, failure_policy=policy
        ),
        tmp_path,
    )
    if policy == "raise":
        with pytest.raises(RuntimeError):
            callback.pre_train()
    else:
        callback.pre_train()
        callback.log_metrics(10, {"train/loss": 1.0})
        callback.post_checkpoint_saved("oss://bucket/run/step10")
        finish(callback)


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_unreachable_server_obeys_failure_policy(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """A real refused HTTP connection is bounded and follows warn/raise policy."""
    import socket

    # Reserve a local port without listening; no remote server or credentials.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        uri = f"http://127.0.0.1:{reserved.getsockname()[1]}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        callback = attach(
            module.OlmoEarthMLflowCallback(
                enabled=True, system_metrics=False, failure_policy=policy
            ),
            tmp_path,
        )
        start = time.monotonic()
        if policy == "raise":
            with pytest.raises(RuntimeError):
                callback.pre_train()
        else:
            callback.pre_train()
            callback.log_metrics(10, {"train/loss": 1.0})
            finish(callback)
        assert time.monotonic() - start < 10


@pytest.mark.parametrize(
    "wandb_enabled,mlflow_enabled", [(False, True), (True, False), (True, True)]
)
def test_wandb_and_mlflow_log_independently(
    module: Any,
    tracking: Any,
    tmp_path: Path,
    wandb_enabled: bool,
    mlflow_enabled: bool,
) -> None:
    """Core's existing W&B callback receives unchanged metrics in either mode."""
    from olmo_core.train.callbacks.wandb import WandBCallback

    logged: list[tuple[int, dict[str, float]]] = []
    wandb = WandBCallback(enabled=wandb_enabled)
    wandb._wandb = SimpleNamespace(
        log=lambda metrics, step: logged.append((step, dict(metrics)))
    )
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=mlflow_enabled, system_metrics=False, log_interval=1
        ),
        tmp_path,
    )
    callback.pre_train()
    metrics = {"train/loss": 0.25, "throughput/tokens_per_second": 1024.0}
    for observer in (callback, wandb):
        observer.log_metrics(4, metrics)
    finish(callback)
    assert logged == ([(4, metrics)] if wandb_enabled else [])
    if mlflow_enabled:
        run_id = callback.state_dict()["run_id"]
        run = tracking.client.get_run(run_id)
        assert run.data.metrics["train/loss"] == 0.25
        assert run.data.metrics["throughput/tokens_per_second"] == 1024.0


def test_existing_foreign_active_run_is_not_closed(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """An already-active fluent run is never adopted or ended accidentally."""
    experiment_id = tracking.client.create_experiment("caller-owned")
    foreign = tracking.mlflow.start_run(experiment_id=experiment_id)
    foreign_id = foreign.info.run_id
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, failure_policy="warn"
        ),
        tmp_path,
    )
    callback.pre_train()
    finish(callback)
    assert tracking.mlflow.active_run() is not None
    assert tracking.mlflow.active_run().info.run_id == foreign_id
    assert tracking.client.get_run(foreign_id).info.status == "RUNNING"


def test_training_error_marks_owned_run_failed(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """Trainer errors finalize the tracking run without masking the root error."""
    callback = attach(
        module.OlmoEarthMLflowCallback(enabled=True, system_metrics=False), tmp_path
    )
    callback.pre_train()
    run_id = callback.state_dict()["run_id"]
    callback.on_error(RuntimeError("original training failure"))
    callback.close()
    assert tracking.client.get_run(run_id).info.status == "FAILED"


def test_core_config_roundtrip_and_callback_composition(
    module: Any, tmp_path: Path
) -> None:
    """The callback supports core's CLI merge, JSON config, and builder chaining."""
    from olmo_core.train import TrainerConfig
    from olmo_core.train.callbacks.callback import Callback

    original = TrainerConfig(save_folder=str(tmp_path)).with_callback(
        "mlflow", module.OlmoEarthMLflowCallback()
    )
    configured = original.merge(
        [
            "callbacks.mlflow.enabled=true",
            "callbacks.mlflow.experiment_name=nano-validation",
            "callbacks.mlflow.run_name=nano-one",
            "callbacks.mlflow.log_interval=3",
            "callbacks.mlflow.dataset_num_samples=42",
        ]
    )
    exported = json.loads(json.dumps(configured.as_config_dict()))
    restored = TrainerConfig.from_dict(exported)
    callback = restored.callbacks["mlflow"]
    assert isinstance(callback, module.OlmoEarthMLflowCallback)
    assert callback.enabled is True
    assert callback.experiment_name == "nano-validation"
    assert callback.run_name == "nano-one"
    assert callback.log_interval == 3
    assert callback.dataset_num_samples == 42
    assert "_client" not in exported["callbacks"]["mlflow"]
    assert original.callbacks["mlflow"].enabled is False
    composed = restored.with_callback("another_observer", Callback())
    assert "another_observer" in composed.callbacks
    assert composed.callbacks["mlflow"] is not callback


def test_core_trainer_checkpoint_state_roundtrip(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """Core's actual state_dict/load_state_dict transport the MLflow identity."""
    from io import BytesIO

    import torch
    from olmo_core.train import Trainer

    callback = attach(
        module.OlmoEarthMLflowCallback(enabled=True, system_metrics=False), tmp_path
    )
    callback.pre_train()
    saved_data = {"cursor": 30}
    source = SimpleNamespace(
        global_step=30,
        global_train_tokens_seen=300,
        global_train_petaflops=0.0,
        max_steps=100,
        data_loader=SimpleNamespace(state_dict=lambda: saved_data),
        epoch=1,
        callbacks={"mlflow": callback},
    )
    payload = BytesIO()
    torch.save(Trainer.state_dict(source), payload)
    payload.seek(0)
    loaded = torch.load(payload, weights_only=False)
    finish(callback)
    restored = attach(
        module.OlmoEarthMLflowCallback(enabled=True, system_metrics=False), tmp_path
    )
    received = []
    target = SimpleNamespace(
        data_loader=SimpleNamespace(load_state_dict=received.append),
        callbacks={"mlflow": restored},
    )
    Trainer.load_state_dict(target, loaded)
    restored.trainer.global_step = target.global_step
    restored.pre_train()
    assert target.global_step == 30
    assert received == [saved_data]
    assert restored.run_id == callback.run_id
    finish(restored)


def test_eval_metrics_and_checkpoint_step_bypass_sampling(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """Evaluation's sparse points survive even between regular log intervals."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, log_interval=10
        ),
        tmp_path,
    )
    callback.pre_train()
    callback.log_metrics(
        7,
        {
            "train/loss": 1.1,
            "eval/classification/accuracy": 0.92,
            "checkpoint_step": 7,
        },
    )
    run = tracking.client.get_run(callback.run_id)
    assert "train/loss" not in run.data.metrics
    assert run.data.metrics["eval/classification/accuracy"] == 0.92
    assert run.data.metrics["checkpoint_step"] == 7
    assert (
        tracking.client.get_metric_history(
            callback.run_id, "eval/classification/accuracy"
        )[0].step
        == 7
    )
    finish(callback)
    assert tracking.client.get_run(callback.run_id).data.metrics["train/loss"] == 1.1


def test_resume_preserves_parameters_and_saves_updated_config(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """Immutable MLflow params remain original while resume overrides stay inspectable."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            system_metrics=False,
            config={"init_seed": 12, "trainer": {"max_steps": 10}},
        ),
        tmp_path,
    )
    callback.pre_train()
    state = callback.state_dict()
    finish(callback)
    resumed = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            system_metrics=False,
            config={
                "init_seed": 12,
                "trainer": {"max_steps": 20},
                "new_note": "continued",
            },
        ),
        tmp_path,
        step=10,
    )
    resumed.load_state_dict(state)
    resumed.pre_train()
    finish(resumed)
    run = tracking.client.get_run(callback.run_id)
    assert resumed.run_id == callback.run_id
    assert run.data.params["trainer.max_steps"] == "10"
    assert run.data.params["new_note"] == "continued"
    latest = json.loads(
        Path(
            tracking.client.download_artifacts(callback.run_id, "config/step10.json")
        ).read_text()
    )
    assert latest["trainer"]["max_steps"] == 20


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_resume_rejects_conflicting_run_id(
    module: Any, tmp_path: Path, policy: str
) -> None:
    """A configured different run cannot silently fork the checkpoint's identity."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True,
            run_id="configured-id",
            system_metrics=False,
            failure_policy=policy,
        ),
        tmp_path,
    )
    state = {"run_id": "checkpoint-id", "tracking_uri": "http://127.0.0.1:5000"}
    if policy == "raise":
        with pytest.raises(RuntimeError):
            callback.load_state_dict(state)
    else:
        callback.load_state_dict(state)
        callback.pre_train()  # disabled reporting makes no network request
    assert callback.state_dict() == state
    callback.on_error(RuntimeError("configuration conflict"))
    finish(callback)


def test_resume_rejects_different_server(
    module: Any, tracking: Any, tmp_path: Path
) -> None:
    """A saved run is not looked up on an unrelated tracking server."""
    callback = attach(
        module.OlmoEarthMLflowCallback(
            enabled=True, system_metrics=False, failure_policy="raise"
        ),
        tmp_path,
    )
    callback.load_state_dict(
        {"run_id": "existing-id", "tracking_uri": "https://other.invalid/mlflow"}
    )
    with pytest.raises(RuntimeError):
        callback.pre_train()
    callback.on_error(RuntimeError("server mismatch"))
    callback.close()
