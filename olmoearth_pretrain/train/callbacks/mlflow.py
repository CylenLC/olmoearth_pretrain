"""Optional rank-zero MLflow tracking using OLMo-core's callback lifecycle.

Only metadata is uploaded. Dataset contents and checkpoint tensors are never read
by this callback. MLflow is imported only when tracking is enabled on rank zero.
"""

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks.callback import Callback

logger = logging.getLogger(__name__)


def _sensitive(key: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", key.lower())
    return (
        any(
            word in compact
            for word in (
                "accesskey",
                "apikey",
                "secret",
                "password",
                "passwd",
                "credential",
                "authorization",
                "authtoken",
                "privatekey",
            )
        )
        or compact in {"token", "env", "environment", "envvars", "environ"}
        or compact.endswith("token")
    )


def _clean_string(value: str) -> str:
    def clean_url(match: re.Match) -> str:
        try:
            parts = urlsplit(match.group())
            return urlunsplit(
                (parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", "")
            )
        except ValueError:
            return "[REDACTED URL]"

    value = re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+", clean_url, value)
    # Configs sometimes embed shell assignments instead of structured fields.
    value = re.sub(
        r"(?i)((?:[\w-]*(?:token|password|passwd|secret|access[_-]?key|api[_-]?key)[\w-]*)\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?i)(--[\w-]*(?:token|password|passwd|secret|access[_-]?key|api[_-]?key)[\w-]*\s+)(?:\"[^\"]*\"|'[^']*'|\S+)",
        r"\1[REDACTED]",
        value,
    )
    return re.sub(r"(?i)\b(Bearer|Basic)\s+\S+", r"\1 [REDACTED]", value)


def sanitize(value: Any) -> Any:
    """Return JSON-safe metadata with credentials and signed URL queries removed."""
    if isinstance(value, Mapping):
        # Support name/value forms used by environment and secret configurations.
        if any(_sensitive(str(value.get(key, ""))) for key in ("name", "key")):
            return {
                str(k): "[REDACTED]"
                if k == "value" or _sensitive(str(k))
                else sanitize(v)
                for k, v in value.items()
            }
        return {
            str(k): "[REDACTED]" if _sensitive(str(k)) else sanitize(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (str, Path)):
        return _clean_string(str(value))
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    # Do not serialize arbitrary reprs, which can contain credentials.
    return f"<{type(value).__name__}>"


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, str]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[_metric_key(name)] = (
                json.dumps(item, ensure_ascii=False)
                if not isinstance(item, str)
                else item
            )[:500]
    return result


def _metric_key(key: str) -> str:
    # Preserve upstream names where MLflow accepts them, disambiguate otherwise.
    clean = re.sub(r"[^\w/ .-]", "_", key)[:230]
    if clean != key:
        clean += "_" + hashlib.sha256(key.encode()).hexdigest()[:12]
    return clean


@dataclass
class OlmoEarthMLflowCallback(Callback):
    """Track existing Trainer metrics without changing training computations.

    ``failure_policy='warn'`` disables reporting after a failure; ``'raise'``
    propagates it. HTTP requests use short timeouts and no retries by default,
    overridable with MLflow's own environment variables. Metrics run on the
    Trainer's existing bookkeeping thread when async bookkeeping is enabled.
    """

    priority: ClassVar[int] = 4  # Before initial checkpoint saves and evaluation.
    enabled: bool = False
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "olmoearth"
    run_name: str | None = None
    run_id: str | None = None
    log_interval: int = 10
    system_metrics: bool = True
    system_metrics_interval: float = 10.0
    failure_policy: str = "warn"
    evaluation_only: bool = False
    config: dict[str, Any] | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    dataset_uri: str | None = None
    dataset_num_samples: int | None = None
    dataset_preprocessing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep runtime state outside the serializable callback configuration."""
        self._client = None
        self._mlflow = None
        self._monitor = None
        self._disabled = False
        self._owns_run = False
        self._completed = False
        self._failed = False
        self._pending_error = None
        self._closed = False
        self._resolved_uri = None
        self._resume_uri = None
        self._last_metrics = None
        self._last_logged_step = None
        self._metrics_by_step = OrderedDict()
        self._checkpoint_metrics = {}
        # TrainerConfig.with_callback() deep-copies callbacks before attaching
        # them. Create the unpicklable lock only once training actually starts.
        self._lock = None

    def _get_lock(self):
        if self._lock is None:
            self._lock = threading.RLock()
        return self._lock

    def _can_log(self) -> bool:
        return (
            self.enabled and get_rank() == 0 and not self._disabled and not self._closed
        )

    def _guard(self, operation: Callable[[], Any]) -> None:
        if not self._can_log():
            return
        try:
            with self._get_lock():
                if self._can_log():
                    operation()
        except Exception as exc:  # noqa: BLE001 - optional telemetry must fail open
            self._disabled = True
            self._pending_error = type(exc).__name__
            self._stop_monitor()
            # Never log exception bodies: HTTP errors may contain credentials.
            logger.warning("MLflow reporting disabled after %s", type(exc).__name__)
            if self.failure_policy == "raise":
                raise RuntimeError(
                    f"MLflow reporting failed ({type(exc).__name__})"
                ) from None

    def state_dict(self) -> dict[str, Any]:
        """Persist identity through the existing Trainer checkpoint state."""
        if self.enabled and get_rank() == 0:
            with self._get_lock():
                if self._trainer is not None and self._client is not None:
                    self._checkpoint_metrics[self.step] = dict(
                        self._metrics_by_step.get(self.step, {})
                    )
                return {
                    "run_id": self.run_id,
                    "tracking_uri": sanitize(self._resume_uri or self._resolved_uri),
                }
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Resume the checkpoint's run; an explicit different ID is an error."""
        if not self.enabled or get_rank() != 0:
            return

        def restore() -> None:
            saved_id = state_dict.get("run_id")
            if saved_id:
                conflict = self.run_id and self.run_id != saved_id
                self.run_id = saved_id
                self._resume_uri = state_dict.get("tracking_uri")
                if conflict:
                    raise ValueError("Configured MLflow run_id differs from checkpoint")

        self._guard(restore)

    def pre_train(self) -> None:
        """Create or resume a run on global rank zero, before checkpointing."""
        self._guard(self._start)

    def _start(self) -> None:
        if self._client is not None:
            return
        if self.failure_policy not in {"warn", "raise"}:
            raise ValueError("failure_policy must be warn or raise")
        if self.log_interval < 1 or self.system_metrics_interval <= 0:
            raise ValueError("MLflow logging intervals must be positive")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "5")
        import mlflow
        from mlflow import MlflowClient

        self._mlflow = mlflow
        self._resolved_uri = os.environ.get("MLFLOW_TRACKING_URI") or self.tracking_uri
        if self._resume_uri and sanitize(self._resolved_uri) != self._resume_uri:
            raise ValueError(
                "Checkpoint MLflow tracking server differs from configuration"
            )
        self._client = MlflowClient(tracking_uri=self._resolved_uri)
        if self.evaluation_only:
            if not self.run_id:
                raise ValueError("evaluation_only requires the training run_id")
            self._client.get_run(self.run_id)
            clean = dict(self.config or {})
            clean.pop("launch", None)
            self._artifact(clean, f"evaluation/config-{time.time_ns()}.json")
            return
        if mlflow.active_run() is not None:
            raise RuntimeError("An unrelated MLflow run is already active")
        mlflow.set_tracking_uri(self._resolved_uri)
        resumed = bool(self.run_id)
        if not self.run_id:
            experiment = self._client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                try:
                    experiment_id = self._client.create_experiment(self.experiment_name)
                except Exception:
                    # Another training job may have created this experiment first.
                    experiment = self._client.get_experiment_by_name(
                        self.experiment_name
                    )
                    if experiment is None:
                        raise
                    experiment_id = experiment.experiment_id
            else:
                experiment_id = experiment.experiment_id
            run = self._client.create_run(
                experiment_id, run_name=sanitize(self.run_name)
            )
            self.run_id = run.info.run_id
        else:
            self._client.update_run(self.run_id, status="RUNNING")
        # Explicit client identity avoids context providers and fluent active-run
        # ownership. Own the built-in monitor so it always stops even if ending
        # a run fails while the tracking server is unavailable.
        self._owns_run = True
        if self.system_metrics:
            from mlflow.system_metrics.system_metrics_monitor import (
                SystemMetricsMonitor,
            )

            self._monitor = SystemMetricsMonitor(
                self.run_id,
                sampling_interval=self.system_metrics_interval,
                resume_logging=resumed,
            )
            self._monitor.start()
        self._record_configuration()
        self._record_dataset()
        logger.info("MLflow run ID: %s", self.run_id)

    def _artifact(self, value: Any, path: str) -> None:
        self._client.log_dict(self.run_id, sanitize(value), path)

    def _record_configuration(self) -> None:
        from mlflow.entities import Param

        clean = sanitize(self.config or {})
        # Launch recipes can contain opaque shell commands and secret mappings.
        clean.pop("launch", None)
        self._artifact(clean, f"config/step{self.step}.json")
        params = _flatten(clean)
        previous = self._client.get_run(self.run_id).data.params
        # MLflow parameters are immutable. Preserve initial params on resume,
        # retaining the full updated configuration in the per-step artifact.
        new = [Param(k, v) for k, v in params.items() if k not in previous]
        for offset in range(0, len(new), 100):
            self._client.log_batch(self.run_id, params=new[offset : offset + 100])
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {},
        }
        for package in (
            "olmoearth-pretrain",
            "ai2-olmo-core",
            "torch",
            "mlflow-skinny",
            "psutil",
            "nvidia-ml-py",
        ):
            try:
                environment["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            environment["git_commit"] = sha
            if (
                "mlflow.source.git.commit"
                not in self._client.get_run(self.run_id).data.tags
            ):
                self._client.set_tag(self.run_id, "mlflow.source.git.commit", sha)
        except (OSError, subprocess.SubprocessError):
            pass
        self._artifact(environment, f"environment/step{self.step}.json")

    def _record_dataset(self) -> None:
        if not self.dataset_name and not self.dataset_uri:
            return
        from mlflow.entities import Dataset, DatasetInput, InputTag

        metadata = sanitize(
            {
                "name": self.dataset_name or "OlmoEarth",
                "version": self.dataset_version,
                "uri": self.dataset_uri,
                "num_samples": self.dataset_num_samples,
                "preprocessing": self.dataset_preprocessing,
            }
        )
        # This is an identity digest of metadata, never a hash of dataset bytes.
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:32]
        dataset = Dataset(
            name=metadata["name"],
            digest=digest,
            source_type="olmoearth",
            source=json.dumps({"uri": metadata["uri"], "version": metadata["version"]}),
            profile=json.dumps({"num_rows": metadata["num_samples"]}),
        )
        self._client.log_inputs(
            self.run_id,
            [DatasetInput(dataset, [InputTag("mlflow.data.context", "training")])],
        )
        self._artifact(metadata, f"dataset/{digest}.json")

    def _write_metrics(self, step: int, metrics: dict[str, float]) -> None:
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if not _sensitive(k) and isinstance(v, (float, int)) and math.isfinite(v)
        }
        clean["global_step"] = float(step)
        batch = [Metric(_metric_key(k), v, timestamp, step) for k, v in clean.items()]
        for offset in range(0, len(batch), 1000):
            self._client.log_batch(
                self.run_id, metrics=batch[offset : offset + 1000], synchronous=True
            )

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Forward reduced metrics at their original step, without mutation."""

        def record() -> None:
            if self._client is None:
                return
            self._metrics_by_step.setdefault(step, {}).update(metrics)
            while len(self._metrics_by_step) > 128:
                self._metrics_by_step.popitem(last=False)
            self._last_metrics = (step, dict(self._metrics_by_step[step]))
            if step % self.log_interval == 0 or step == 1:
                selected = metrics
                self._last_logged_step = step
            else:
                selected = {
                    k: v
                    for k, v in metrics.items()
                    if k.startswith(("eval", "downstream/")) or k == "checkpoint_step"
                }
            if selected:
                self._write_metrics(step, selected)

        self._guard(record)

    def post_checkpoint_saved(self, path: str | Path) -> None:
        """Log a URI manifest after a successful save, including async saves."""

        def record() -> None:
            if self._client is None or self.evaluation_only:
                return
            # A completed asynchronous checkpoint can lag the Trainer's step.
            match = re.fullmatch(
                r"step(\d+)", urlsplit(str(path)).path.rstrip("/").rsplit("/", 1)[-1]
            )
            if match is None:
                raise ValueError("Unrecognized OLMo-core checkpoint directory")
            step = int(match.group(1))
            metrics = self._checkpoint_metrics.pop(
                step, self._metrics_by_step.get(step, {})
            )
            self._artifact(
                {
                    "uri": str(path),
                    "step": step,
                    "run_id": self.run_id,
                    "metrics": metrics,
                },
                f"checkpoints/step{step}.json",
            )
            self._client.set_tag(
                self.run_id, "olmoearth.checkpoint_uri", sanitize(str(path))
            )
            self._write_metrics(step, {"checkpoint/step": step})

        self._guard(record)

    def log_evaluation(
        self,
        name: str,
        metrics: dict[str, float],
        config: dict | None = None,
        results: dict | None = None,
        step: int | None = None,
    ) -> None:
        """Attach existing downstream results; never run evaluation algorithms."""

        def record() -> None:
            if self._client is None:
                return
            eval_step = self.step if step is None else step
            safe_name = re.sub(r"[^\w.-]", "_", name).strip(".")[:100] or "evaluation"
            safe_name += "-" + hashlib.sha256(name.encode()).hexdigest()[:8]
            self._artifact(
                {
                    "name": name,
                    "step": eval_step,
                    "metrics": metrics,
                    "config": config,
                    "results": results,
                },
                f"evaluation/{safe_name}/step{eval_step}-{time.time_ns()}.json",
            )
            if metrics:
                self._write_metrics(eval_step, metrics)

        self._guard(record)

    def post_step(self) -> None:
        """Surface strict errors from async checkpoint notifications to training."""
        if (
            self.enabled
            and get_rank() == 0
            and self.failure_policy == "raise"
            and self._pending_error
            and not self._failed
        ):
            raise RuntimeError(f"MLflow reporting failed ({self._pending_error})")

    def post_train(self) -> None:
        """Defer termination until the checkpointer and bookkeeping have closed."""
        self._completed = True

    def on_error(self, exc: BaseException) -> None:
        """Mark failure without exposing exception text to the tracking server."""
        self._failed = True

    def close(self) -> None:
        """Flush the final sampled step and stop MLflow's built-in monitoring."""
        if not self.enabled or get_rank() != 0 or self._closed:
            return
        with self._get_lock():
            self._close()

    def _stop_monitor(self) -> None:
        if self._monitor is not None:
            try:
                self._monitor.finish()
            except Exception as exc:  # noqa: BLE001 - best effort telemetry cleanup
                logger.warning("MLflow monitor cleanup failed (%s)", type(exc).__name__)
            finally:
                self._monitor = None

    def _close(self) -> None:
        try:
            if self._last_metrics and self._last_metrics[0] != self._last_logged_step:
                self._guard(lambda: self._write_metrics(*self._last_metrics))
        finally:
            self._stop_monitor()
            try:
                if self._owns_run:
                    status = (
                        "FAILED"
                        if self._failed or self._disabled or not self._completed
                        else "FINISHED"
                    )
                    self._client.set_terminated(self.run_id, status=status)
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask training errors
                logger.warning("MLflow cleanup failed (%s)", type(exc).__name__)
                if self.failure_policy == "raise" and not self._failed:
                    raise RuntimeError("MLflow cleanup failed") from None
            finally:
                self._closed = True
        self.post_step()
