# MLflow tracking

Based on official upstream/main `6583a536c428ad004bb3c223add0576087d67e92`.
OLMo-core remains 2.6.0 and locked PyTorch remains 2.9.1. The client extra is
`mlflow-skinny>=2.22.5,<3`, psutil and nvidia-ml-py; no server is deployed here.

## Install and connect

In an existing compatible training environment:

```bash
pip install -e '.[mlflow]'
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
curl --fail "$MLFLOW_TRACKING_URI/health"
```

For a separate ECS host, run an SSH tunnel on the training host (substitute your
SSH destination), then use the same URI:

```bash
ssh -N -L 5000:127.0.0.1:5000 USER@ECS_HOST
```

These are connection templates; the user's ECS endpoint was not contacted.
For a fresh environment use `uv sync --locked --extra training --extra beaker
--extra eval --extra mlflow`. Official recipe imports require eval dependencies
including when evaluation is disabled. The local validation environment needed
`setuptools<81` for upstream class-registry's pkg_resources import.

## Nano smoke and resume

From the repository root, on a CUDA training machine:

```bash
python scripts/tools/mlflow_nano_smoke.py --work-dir /tmp/nano-mlflow --mlflow --steps 2
python scripts/tools/mlflow_nano_smoke.py --work-dir /tmp/nano-mlflow --mlflow --steps 3
```

The second command restores the existing Trainer checkpoint and its MLflow Run
ID automatically. This helper generates four synthetic HDF5 samples and invokes
`scripts/official/v1_2/nano.py train_single`; it uses small crops, a single
modality and patch size 1, with the existing model/loss/optimizer implementations.
It disables W&B and downstream evaluation. Omit `--mlflow` for a disabled test.
This command was attempted locally: official training reaches CUDA memory
monitoring and fails on this CPU-only Mac. A successful training/resume smoke on
CUDA remains required.

For real data, use the official entrypoint and existing recipe overrides:

```bash
python scripts/official/v1_2/nano.py train_single nano-mlflow local \
  --common.save_folder=/data/checkpoints/nano-mlflow \
  --dataset.h5py_dir=/data/olmoearth-h5 \
  --trainer.callbacks.wandb.enabled=false \
  --trainer.callbacks.mlflow.enabled=true \
  --trainer.callbacks.mlflow.experiment_name=olmoearth \
  --trainer.callbacks.mlflow.run_name=nano-mlflow \
  --trainer.callbacks.mlflow.dataset_uri=oss://YOUR_BUCKET/DATASET \
  --trainer.callbacks.mlflow.dataset_version=YOUR_VERSION
```

Replace data paths with an existing compatible dataset and apply your normal
batch/resource overrides. OSS URI here is metadata, not a new storage adapter.
Existing checkpoint storage support is unchanged. To restore elsewhere use
`--trainer.load_path=CHECKPOINT_PATH`; retain trainer state loading. For legacy
checkpoints without MLflow state supply `--trainer.callbacks.mlflow.run_id=RUN_ID`.
Use the same tracking server. A conflicting explicit ID or server disables
tracking in warn mode (preserving the checkpoint identity), or raises in strict mode.

## Configuration

All fields are under `trainer.callbacks.mlflow`:

| Field | Default | Meaning |
|---|---|---|
| enabled | false | Optional client, imported only on global rank 0 |
| tracking_uri | http://127.0.0.1:5000 | MLFLOW_TRACKING_URI takes precedence |
| experiment_name | olmoearth | Create/reuse experiment |
| run_name / run_id | null | Name defaults to experiment run name; optional existing ID |
| log_interval | 10 | First step, every N steps, final step; eval metrics bypass sampling |
| system_metrics | true | MLflow built-in CPU/memory/GPU monitor, rank-zero host only |
| system_metrics_interval | 10 seconds | MLflow sampling environment variable can override |
| failure_policy | warn | Disable reporting on failure; raise opts into training failure |
| evaluation_only | false in training, true in eval | Attach to training ID without ending it |
| dataset_name / dataset_version / dataset_uri | inferred / null / h5py_dir | Metadata only |
| dataset_num_samples | prepared dataset length | No additional scan |
| dataset_preprocessing | dataset + loader config | Optional explicit metadata |

Existing loss, LR, gradient norm, throughput and other numeric Trainer metrics
retain their step. Unsupported MLflow name characters are normalized with a
hash suffix. Nonfinite metrics are omitted. System metric steps are MLflow's
sampling index, not training steps. W&B remains independently configurable.

Config snapshots, package/Python/platform information, Git SHA, seeds and
flattened parameters are recorded. Resumes preserve immutable initial params
and save updated config artifacts. Secrets are redacted; launch recipes and
ambient environment variables are excluded. Do not place credentials in free
text metadata or dataset names.

DatasetInput uses a metadata digest (not a content checksum). Checkpoint hooks
write only small JSON URI/step/metric manifests, never model weights. Async
checkpoint steps come from stepN paths. Run IDs are saved through the existing
Callback state inside Trainer checkpoints. No core checkpoint format is replaced.

In-loop downstream results/configuration are attached to the training run.
Standalone evaluate and checkpoint sweep commands accept the same flags plus
`--trainer.callbacks.mlflow.run_id=TRAINING_RUN_ID`. Evaluation-only mode does
not change training run status. Beaker eval jobs require explicit extra overrides,
client installation and network connectivity; localhost refers to that job host.

HTTP defaults are 3 seconds and zero retries; artifact timeout is 5 seconds.
Existing MLFLOW_HTTP_REQUEST_TIMEOUT, MLFLOW_HTTP_REQUEST_MAX_RETRIES and
MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT values take precedence. Reporting uses
Trainer bookkeeping, with bounded sampling; requests are synchronous within that
thread. Failure disables reporting for the rest of the process. System monitor
cleanup precedes run termination. Strict async errors surface via post_step/close.

## Validation

```bash
python -m pytest tests_mlflow -q
uv lock --check
ruff check olmoearth_pretrain/train/callbacks/mlflow.py scripts/tools/mlflow_nano_smoke.py tests_mlflow
```

26 tests passed against real MLflow 2.22.5 FileStore and OLMo-core 2.6.0, including
CPU/memory monitoring, dataset inputs, metrics, checkpoint state roundtrip,
resume identity, redaction, HTTP refusal, optional dependency absence, rank
filtering, configuration serialization and W&B observer independence.
Lock check passed. New files pass Ruff; modified upstream files retain six
pre-existing SIM102/TRY004 findings under the installed Ruff rules.

Not verified: successful CUDA Nano training/resume, GPU telemetry, real multi-rank
execution (rank guards are unit-tested), ECS connectivity, real OSS checkpoint
writes, downstream evaluation algorithms, live W&B service, or MLflow 3.x server
compatibility. No model/loss/optimizer/DataLoader or OLMo-core source was changed.

## Review follow-up

The full feature diff was reviewed against the official base. Follow-up fixes
preserve checkpoint identity while tracking is temporarily disabled, redact
structured CLI credentials and case-varied name/value records, use opaque hashed
evaluation artifact directory names, and prevent out-of-order asynchronous saves
from moving the latest checkpoint URI tag backwards within a process.
Regression coverage now contains 30 passing tests, including strict failure
propagation from Future callbacks. Hardware/service validation limits above remain.
