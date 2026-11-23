"""Run the official v1.2 Nano recipe on four small, synthetic local HDF5 samples.

This exercises the existing training loop, data loader, loss, optimizer and
checkpoint implementation. It is a client smoke test, not a quality benchmark.
Requires the official recipe's training, beaker and eval dependencies (it imports
evaluation modules even when evaluation is disabled); add mlflow for --mlflow.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def prepare_fixture(directory: Path) -> None:
    """Create deterministic fixture data once without accessing external storage."""
    if directory.exists():
        return
    directory.mkdir(parents=True)
    rng = np.random.default_rng(42)
    timestamps = np.array([[1, month, 2020] for month in range(12)], dtype=np.int32)
    for index in range(4):
        with h5py.File(directory / f"sample_{index}.h5", "w") as sample:
            sample.create_dataset(
                "sentinel2_l2a",
                data=rng.integers(1, 10000, size=(8, 8, 12, 12), dtype=np.uint16),
            )
            sample.create_dataset("timestamps", data=timestamps)
    (directory / "sample_metadata.csv").write_text(
        "sentinel2_l2a\n1\n1\n1\n1\n", encoding="utf-8"
    )
    np.save(directory / "latlon_distribution.npy", np.zeros((4, 2)))


def main() -> None:
    """Prepare a local fixture and launch the unmodified official Nano entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2, help="Total target global step")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    root = Path(__file__).resolve().parents[2]
    work_dir = args.work_dir.resolve()
    dataset_dir = work_dir / "data" / "sentinel2_l2a" / "4"
    prepare_fixture(dataset_dir)
    command = [
        sys.executable,
        str(root / "scripts/official/v1_2/nano.py"),
        "dry_run" if args.dry_run else "train_single",
        "mlflow-nano-smoke",
        "local",
        f"--common.save_folder={work_dir / 'checkpoints'}",
        "--common.training_modalities=[sentinel2_l2a]",
        "--model.encoder_config.band_dropout_modalities=[sentinel2_l2a]",
        "--model.encoder_config.max_patch_size=1",
        f"--dataset.h5py_dir={dataset_dir}",
        "--data_loader.global_batch_size=2",
        "--data_loader.num_workers=0",
        "--data_loader.min_patch_size=1",
        "--data_loader.max_patch_size=1",
        "--data_loader.sampled_hw_p_list=[4]",
        "--data_loader.tile_size=8",
        "--data_loader.max_timesteps=2",
        "--data_loader.token_budget=1024",
        "--train_module.rank_microbatch_size=2",
        "--trainer.no_evals=true",
        "--trainer.async_bookkeeping=false",
        "--trainer.metrics_collect_interval=1",
        "--trainer.max_duration.unit=steps",
        f"--trainer.max_duration.value={args.steps}",
        "--trainer.callbacks.checkpointer.save_interval=1",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=null",
        "--trainer.callbacks.checkpointer.save_async=false",
        "--trainer.callbacks.downstream_evaluator.enabled=false",
        "--trainer.callbacks.wandb.enabled=false",
        f"--trainer.callbacks.mlflow.enabled={str(args.mlflow).lower()}",
        "--trainer.callbacks.mlflow.experiment_name=olmoearth-nano-smoke",
        "--trainer.callbacks.mlflow.log_interval=1",
        "--trainer.callbacks.mlflow.dataset_name=synthetic-sentinel2",
        "--trainer.callbacks.mlflow.dataset_version=fixture-v1",
        *overrides,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(root), env.get("PYTHONPATH")])
    )
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MPLCONFIGDIR", str(work_dir / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(work_dir / "cache"))
    if args.tracking_uri:
        env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
