"""Bounded, read-only SSD/OSS parity check using the official Dataset and loader.

Run as a module (see docs/oss_dataset.md). Only the specified 3--10 H5 objects
and the two metadata files are read. This is a CPU diagnostic, not training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import psutil
import s3fs.core
import torch
from upath import UPath

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDatasetConfig,
    OSSConfig,
)
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.train.masking import MaskingConfig


def install_transfer_meter() -> None:
    """Count successful range payloads in this diagnostic, including spawn workers."""
    directory = os.environ.get("OLMOEARTH_OSS_METER_DIR")
    if not directory:
        return
    original = s3fs.core._fetch_range

    def measured(fs, bucket, key, version_id, start, end, req_kw=None):
        data = original(fs, bucket, key, version_id, start, end, req_kw)
        with (Path(directory) / f"{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps({"key": key, "bytes": len(data)}) + "\n")
        return data

    s3fs.core._fetch_range = measured


# Spawn imports this module before unpickling the official Dataset. No client,
# connection, or HDF5 handle is created here. Instrumentation is test-only.
install_transfer_meter()


def seed_all(seed: int) -> None:
    """Replay Python, NumPy and Torch random transforms exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def assert_exact(left, right, location: str = "root") -> None:
    """Compare nested official samples without tolerance, preserving NaN positions."""
    if isinstance(left, (np.ndarray, np.generic, torch.Tensor)):
        a = left.detach().cpu().numpy() if isinstance(left, torch.Tensor) else left
        b = right.detach().cpu().numpy() if isinstance(right, torch.Tensor) else right
        assert type(left) is type(right), location
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, location
        # assert_array_equal treats corresponding NaNs as equal, compares Inf
        # signs, and reports the mismatched array positions on failure.
        np.testing.assert_array_equal(a, b, err_msg=location, strict=True)
    elif dataclasses.is_dataclass(left):
        assert type(left) is type(right), location
        for field in dataclasses.fields(left):
            assert_exact(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{location}.{field.name}",
            )
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), location
        for key in left:
            assert_exact(left[key], right[key], f"{location}/{key}")
    elif isinstance(left, tuple) and hasattr(left, "_asdict"):
        assert type(left) is type(right), location
        assert_exact(left._asdict(), right._asdict(), location)
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), location
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{location}[{index}]")
    else:
        assert type(left) is type(right) and left == right, location


def compare_h5(left: h5py.Group, right: h5py.Group, stats: dict) -> None:
    """Recursively compare every group, dataset, attribute and missing-value mask."""
    assert type(left) is type(right), left.name
    assert_exact(dict(left.attrs), dict(right.attrs), left.name + "@attrs")
    if isinstance(left, h5py.Dataset):
        assert left.shape == right.shape and left.dtype == right.dtype, left.name
        a, b = left[()], right[()]
        assert_exact(a, b, left.name)
        if isinstance(a, np.ndarray) and a.dtype.kind in "fc":
            stats["nan_values"] += int(np.isnan(a).sum())
            stats["inf_values"] += int(np.isinf(a).sum())
        stats["datasets"] += 1
        return
    assert left.keys() == right.keys(), left.name
    for key in left:
        compare_h5(left[key], right[key], stats)


@contextmanager
def resources():
    """Sample this process plus its DataLoader children, never inspecting arguments."""
    process = psutil.Process()
    readings: list[tuple[float, int]] = []
    stopped = threading.Event()

    def collect():
        known = {process.pid: process}
        while not stopped.is_set():
            cpu, rss = 0.0, 0
            for item in [process, *process.children(recursive=True)]:
                try:
                    current = known.setdefault(item.pid, item)
                    cpu += current.cpu_percent()
                    rss += current.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            readings.append((cpu, rss))
            stopped.wait(0.1)

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    result = {}
    try:
        yield result
    finally:
        stopped.set()
        thread.join()
        result.update(
            cpu_percent_mean=float(np.mean([x[0] for x in readings[1:]] or [0])),
            rss_peak_bytes=max((x[1] for x in readings), default=0),
        )


def timed_loader(dataset, output: Path, workers: int):
    """Use unchanged sampler, random transform, masking, collator and spawn workers."""
    seed_all(42)
    start = time.perf_counter()
    loader = OlmoEarthDataLoaderConfig(
        work_dir=str(output),
        global_batch_size=1,
        min_patch_size=1,
        max_patch_size=2,
        sampled_hw_p_list=[8, 16],
        seed=42,
        token_budget=10000,
        max_timesteps=4,
        tile_size=128,
        num_workers=workers,
        prefetch_factor=1,
        target_device_type="cpu",
        shuffle=True,
        transform_config=TransformConfig(transform_type="flip_and_rotate"),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
    ).build(dataset)
    loader.reshuffle(epoch=1)
    indices = dataset.sample_indices[loader.get_global_indices()].tolist()
    initialization = time.perf_counter() - start
    result, batches = (
        {
            "workers": workers,
            "initialization_s": initialization,
            "indices": indices,
            "passes": [],
        },
        [],
    )
    with resources() as usage:
        for repeat in range(2):
            loader.reset()
            seed_all(42)
            waits = []
            collected = []
            started = time.perf_counter()
            iterator = iter(loader)
            for _ in range(len(dataset)):
                before = time.perf_counter()
                collected.append(next(iterator))
                waits.append(time.perf_counter() - before)
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise AssertionError("Unexpected extra batch")
            elapsed = time.perf_counter() - started
            assert len(collected) == len(dataset)
            result["passes"].append(
                {
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "samples_per_s": len(dataset) / elapsed,
                    "first_batch_s": waits[0],
                    "batch_wait_s": waits,
                    "p50_s": float(np.percentile(waits, 50)),
                    "p95_s": float(np.percentile(waits, 95)),
                }
            )
            batches.append(collected)
    # Check serialization/resume without changing the official sampler contract.
    state = pickle.loads(pickle.dumps(loader.state_dict()))
    loader.reset()
    loader.load_state_dict(state)
    assert loader.batches_processed == len(dataset)
    result.update(usage)
    return result, batches


def run(args, report: dict) -> None:
    """Verify a bounded sample set; failures propagate and preserve partial evidence."""
    indices = sorted(set(map(int, args.indices.split(","))))
    if not 3 <= len(indices) <= 10:
        raise ValueError("Specify 3--10 unique sample indices")
    local = UPath(args.local_dir)
    if local.protocol != "":
        raise ValueError("The SSD comparison path must be local")
    config = OSSConfig(args.endpoint_url, args.region, args.profile)
    remote = OlmoEarthDatasetConfig(args.oss_dir, [], oss=config).h5py_dir_upath
    total = sum((local / f"sample_{index}.h5").stat().st_size for index in indices)
    if total > 100 * 1024**2:
        raise ValueError("Selected H5 files exceed the 100 MiB diagnostic limit")
    report.update(
        indices=indices,
        selected_h5_bytes=total,
        local_dir=str(local),
        oss_dir=str(remote),
        endpoint_url=args.endpoint_url,
        region=args.region,
        cache_condition="Existing local files and OS caches; no cache clearing. "
        "Fresh HDF5/file objects each call; ordinary s3fs per-file cache.",
    )
    # Compare both required metadata files using the configured UPath handles.
    for name in ("sample_metadata.csv", "latlon_distribution.npy"):
        with (local / name).open("rb") as a, (remote / name).open("rb") as b:
            assert a.read() == b.read(), name
    report["metadata_exact"] = True
    stats = {"datasets": 0, "nan_values": 0, "inf_values": 0}
    sample_details = []
    for index in indices:
        name = f"sample_{index}.h5"
        with (local / name).open("rb") as a, (remote / name).open("rb") as b:
            with h5py.File(a, "r") as left, h5py.File(b, "r") as right:
                compare_h5(left, right, stats)
                sample_details.append(
                    {
                        "name": name,
                        "bytes": (local / name).stat().st_size,
                        "modalities": list(left.keys()),
                    }
                )
    report.update(h5_exact=True, h5_details=sample_details, h5_stats=stats)
    np.save(args.output / "indices.npy", indices)
    import pandas as pd

    with (local / "sample_metadata.csv").open("rb") as stream:
        modalities = [
            key for key in pd.read_csv(stream).columns if key != "sample_index"
        ]
    modalities += ["latlon", "ndvi", "worldcover_onehot"]
    datasets = []
    for root, options in ((local, OSSConfig()), (remote, config)):
        dataset = OlmoEarthDatasetConfig(
            h5py_dir=str(root),
            training_modalities=modalities,
            oss=options,
            filter_idx_file=str(args.output / "indices.npy"),
        ).build()
        dataset.prepare()
        np.testing.assert_array_equal(dataset.sample_indices, indices)
        assert dataset.cache_dir is None
        datasets.append(dataset)
    report["dataset_samples"] = []
    for cutmix in (False, True):
        for index in range(len(indices)):
            values, durations = [], []
            for dataset in datasets:
                dataset.apply_cutmix = cutmix
                seed_all(100 + index)
                started = time.perf_counter()
                values.append(dataset[GetItemArgs(index, 2, 16, target_t=4)])
                durations.append(time.perf_counter() - started)
            assert_exact(*values, f"sample_{indices[index]}:cutmix={cutmix}")
            report["dataset_samples"].append(
                {
                    "index": indices[index],
                    "cutmix": cutmix,
                    "ssd_s": durations[0],
                    "oss_s": durations[1],
                }
            )
    report["dataset_exact"] = True
    for dataset in datasets:
        dataset.apply_cutmix = False
    report["loaders"] = []
    for workers in (0, 1, 2, 4):
        # Alternate paired order. Both runs get identical random states and indices.
        order = (0, 1) if workers in (0, 2) else (1, 0)
        values, metrics = {}, {}
        for source in order:
            name = "ssd" if source == 0 else "oss"
            metrics[name], values[name] = timed_loader(
                datasets[source], args.output / f"{name}-workers-{workers}", workers
            )
        assert metrics["ssd"]["indices"] == metrics["oss"]["indices"]
        assert_exact(values["ssd"], values["oss"], f"loader-workers-{workers}")
        report["loaders"].append(
            {"order": ["ssd" if x == 0 else "oss" for x in order], **metrics}
        )
        print(f"workers={workers}: two passes, exact SSD/OSS batch parity", flush=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
    report["dataloader_exact"] = True


def main() -> None:
    """Parse only non-secret settings; never accept access keys as CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--oss-dir", required=True)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Require a fresh output directory to avoid mixing measurements from old runs.
    args.output.mkdir(parents=True, exist_ok=False)
    meter = args.output.resolve() / "transfer"
    meter.mkdir()
    os.environ["OLMOEARTH_OSS_METER_DIR"] = str(meter)
    install_transfer_meter()
    report = {"status": "RUNNING"}
    try:
        run(args, report)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error_type=type(error).__name__)
        raise
    finally:
        records = [
            json.loads(line)
            for file in meter.glob("*.jsonl")
            for line in file.read_text().splitlines()
        ]
        report["transfer"] = {
            "unique_objects": len({r["key"] for r in records}),
            "range_fetches": len(records),
            "payload_bytes": sum(r["bytes"] for r in records),
            "scope": "Successful s3fs range payloads in parent and spawn workers; "
            "excludes TLS/HTTP overhead, HEAD/LIST bodies and failed retries.",
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps({"status": report["status"], "transfer": report["transfer"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
