"""Measure the actual Nano trainer with bounded fsspec lookahead on or off.

Run from the checkout with that checkout on PYTHONPATH. Reuse a historical
Nano fsspec config.json and use the normal AWS credential provider chain.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from time import monotonic, perf_counter
from types import SimpleNamespace

_output = (
    Path(os.environ["OLMO_NANO_OUTPUT"]) if "OLMO_NANO_OUTPUT" in os.environ else None
)
_thread = threading.local()


class Events:
    """Write numeric timings and counts without logging paths or SDK arguments."""

    def __init__(self, name):
        """Open a per-process event stream."""
        self.stream = (_output / f"{name}-{os.getpid()}.jsonl").open("x", buffering=1)
        self.lock = threading.Lock()

    def write(self, event, **fields):
        """Append one event from a worker, I/O thread, or trainer."""
        row = json.dumps(dict(event=event, time=monotonic(), **fields))
        with self.lock:
            self.stream.write(row + "\n")


def _install_http(events):
    from aiobotocore.endpoint import AioEndpoint
    from aiobotocore.response import StreamingBody

    send = AioEndpoint._send
    read = StreamingBody.read

    async def send_observed(self, request):
        start = perf_counter()
        response = None
        try:
            response = await send(self, request)
            return response
        finally:
            events.write(
                "http",
                method=request.method,
                status=response.status_code if response is not None else None,
                seconds=perf_counter() - start,
            )

    async def read_observed(self, *args, **kwargs):
        data = await read(self, *args, **kwargs)
        events.write("body", bytes=len(data))
        return data

    AioEndpoint._send = send_observed
    StreamingBody.read = read_observed


def _worker_init(worker_id):
    import h5py
    import torch
    from s3fs.core import S3File

    from olmoearth_pretrain.data import dataloader as dl
    from olmoearth_pretrain.data import dataset as ds

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    events = Events("worker")
    _install_http(events)

    wrapper = torch.utils.data.get_worker_info().dataset
    if (
        wrapper.dataset.sec_per_sample is not None
        or wrapper.dataset.cache_dir is not None
    ):
        raise ValueError("The experiment requires uncached, unthrottled sample reads")
    if os.environ["OLMO_NANO_PREFETCH"] == "on":
        download = dl._download_fsspec_payload
        fs = wrapper.dataset.h5py_dir.fs
        fs_open = fs.open

        def fs_open_observed(*args, **kwargs):
            start = perf_counter()
            stream = fs_open(*args, **kwargs)
            _thread.download_open_s = perf_counter() - start
            return stream

        fs.open = fs_open_observed

        def download_observed(path, fs, **kwargs):
            start = perf_counter()
            _thread.download_open_s = 0.0
            payload = download(path, fs, **kwargs)
            elapsed = perf_counter() - start
            open_s = _thread.download_open_s
            events.write(
                "download",
                seconds=elapsed,
                open_s=open_s,
                transfer_s=elapsed - open_s,
                bytes=len(payload),
            )
            return payload

        dl._download_fsspec_payload = download_observed

        submit = dl._OSSGetPrefetcher.submit

        def submit_observed(self, path):
            future = submit(self, path)
            result = future.result

            def result_observed():
                start = perf_counter()
                try:
                    return result()
                finally:
                    _thread.future_wait = perf_counter() - start

            future.result = result_observed
            return future

        dl._OSSGetPrefetcher.submit = submit_observed

    fetch_range = S3File._fetch_range
    open_h5 = ds._open_h5_file
    read_h5 = ds.OlmoEarthDataset.read_h5_file
    read_dataset = h5py.Dataset.__getitem__
    h5_init = h5py.File.__init__
    sample_item = ds.OlmoEarthDataset.__getitem__

    def stage(name, original):
        @wraps(original)
        def timed(*args, **kwargs):
            sample = getattr(_thread, "sample", None)
            if sample is None:
                return original(*args, **kwargs)
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                setattr(sample, name, getattr(sample, name) + perf_counter() - start)

        return timed

    @contextmanager
    def open_observed(path, **kwargs):
        start = perf_counter()
        with open_h5(path, **kwargs) as stream:
            _thread.sample.open_s += perf_counter() - start
            yield stream

    def read_dataset_observed(self, selection):
        sample = getattr(_thread, "sample", None)
        if sample is None:
            return read_dataset(self, selection)
        start, prior_io = perf_counter(), sample.range_s
        try:
            return read_dataset(self, selection)
        finally:
            sample.h5_read_s += max(
                0, perf_counter() - start - (sample.range_s - prior_io)
            )

    def item_observed(self, args):
        sample = _thread.sample = SimpleNamespace(
            open_s=0.0,
            range_s=0.0,
            read_h5_s=0.0,
            h5_read_s=0.0,
            h5_open_s=0.0,
            construct_s=0.0,
            crop_s=0.0,
        )
        start = perf_counter()
        try:
            result = sample_item(self, args)
            # Hash only indices separately in the batch path.
            elapsed = perf_counter() - start
            events.write(
                "sample",
                worker=worker_id,
                index=int(args.idx),
                seconds=elapsed,
                future_wait_s=getattr(_thread, "future_wait", 0.0),
                crop_normalize_s=max(
                    0, elapsed - sample.read_h5_s - sample.construct_s
                ),
                **vars(sample),
            )
            return result
        finally:
            del _thread.sample
            _thread.future_wait = 0.0

    # Include timestamp trimming, padding, missing-value placement and the first
    # sample construction in the construction stage.
    ds.OlmoEarthDataset._crop_timestamps_and_masks = staticmethod(
        stage("construct_s", ds.OlmoEarthDataset._crop_timestamps_and_masks)
    )
    for method in ("_pad_timestamps", "fill_sample_with_missing_values"):
        setattr(
            ds.OlmoEarthDataset,
            method,
            stage("construct_s", getattr(ds.OlmoEarthDataset, method)),
        )
    ds.subset_sample_default = stage("crop_s", ds.subset_sample_default)
    ds.subset_sample_cutmix = stage("crop_s", ds.subset_sample_cutmix)
    ds._open_h5_file = open_observed
    ds.OlmoEarthDataset.read_h5_file = stage("read_h5_s", read_h5)
    ds.OlmoEarthDataset.__getitem__ = item_observed
    h5py.Dataset.__getitem__ = read_dataset_observed
    h5py.File.__init__ = stage("h5_open_s", h5_init)
    S3File._fetch_range = stage("range_s", fetch_range)

    # Compare batch assignment without hashing full tensors on the timing path.
    params = wrapper._get_batch_item_params_iterator

    def observed_params(*args, **kwargs):
        digest = hashlib.sha256()
        count = 0
        for item in params(*args, **kwargs):
            digest.update(int(item[0]).to_bytes(8, "little", signed=True))
            count += 1
            if count == wrapper.data_loader.rank_batch_size:
                events.write(
                    "batch_indices", worker=worker_id, digest=digest.hexdigest()
                )
                digest = hashlib.sha256()
                count = 0
            yield item

    wrapper._get_batch_item_params_iterator = observed_params


def _train(config_path, steps):
    import torch
    from olmo_core.train import LoadStrategy, prepare_training_environment
    from olmo_core.train.common import Duration

    from olmoearth_pretrain.data import dataloader
    from olmoearth_pretrain.internal.experiment import OlmoEarthExperimentConfig, train
    from olmoearth_pretrain.train.callbacks.speed_monitor import (
        OlmoEarthSpeedMonitorCallback,
    )

    dataloader._worker_ignore_sigterm = _worker_init
    config = OlmoEarthExperimentConfig.from_dict(json.loads(config_path.read_text()))
    # LossConfig.build() pops the type before the historical config is saved.
    config.train_module.loss_config.loss_config.setdefault(
        "type", "modality_patch_discrimination_masked_negatives_vec"
    )
    config.train_module.contrastive_config.loss_config.setdefault("type", "InfoNCE")
    if (
        config.data_loader.num_workers != 16
        or config.data_loader.global_batch_size != 512
    ):
        raise ValueError("Expected the historical single-GPU Nano configuration")
    config.dataset.oss.use_fsspec = True
    config.data_loader.max_inflight_oss_gets_per_worker = (
        2 if os.environ["OLMO_NANO_PREFETCH"] == "on" else 1
    )
    config.data_loader.work_dir = str(_output / "work")
    config.trainer.save_folder = str(_output / "work")
    config.trainer.work_dir = str(_output / "work")
    config.trainer.load_path = None
    config.trainer.load_strategy = LoadStrategy.never
    config.trainer.no_checkpoints = True
    config.trainer.no_evals = True
    config.trainer.callbacks["wandb"].enabled = False
    config.trainer.max_duration = Duration.steps(steps)
    events = Events("trainer")
    _install_http(Events("setup"))

    callback = OlmoEarthSpeedMonitorCallback
    pre_load_batch, pre_step, post_step = (
        callback.pre_load_batch,
        callback.pre_step,
        callback.post_step,
    )
    iterator_type = torch.utils.data.dataloader._MultiProcessingDataLoaderIter
    get_data = iterator_type._get_data

    def get_observed(self):
        task, result = get_data(self)
        if task != self._rcvd_idx:
            events.write("out_of_order", task=task, wanted=self._rcvd_idx)
        return task, result

    def before_load(self):
        events.write("load_start", step=self.trainer.global_step + 1)
        return pre_load_batch(self)

    def after_load(self, batch):
        events.write(
            "batch", step=self.trainer.global_step, samples=len(batch[1].timestamps)
        )
        return pre_step(self, batch)

    def after_step(self):
        events.write("step", step=self.trainer.global_step)
        return post_step(self)

    iterator_type._get_data = get_observed
    callback.pre_load_batch = before_load
    callback.pre_step = after_load
    callback.post_step = after_step
    prepare_training_environment()
    train(config)


def _gpu():
    query = ["nvidia-smi", "--format=csv,noheader,nounits"]
    values = subprocess.check_output(
        [*query, "--query-gpu=utilization.gpu,memory.used"], text=True
    ).splitlines()
    return [tuple(map(int, line.split(","))) for line in values]


def _launch(args):
    import psutil

    if _gpu() != [(0, 0)]:
        raise SystemExit("Expected one idle GPU before launching")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if apps.strip():
        raise SystemExit("GPU has an active compute process")
    _output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "OLMO_NANO_PREFETCH": args.prefetch,
        "OLMO_NANO_OUTPUT": str(_output),
    }
    resources = Events("resources")
    with (_output / "train.log").open("wb") as log:
        child = subprocess.Popen(
            [sys.executable, __file__, *sys.argv[1:], "--child"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        secrets = [
            env[key].encode()
            for key in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            )
            if env.get(key)
        ]

        def drain_log():
            for line in child.stdout:
                for secret in secrets:
                    line = line.replace(secret, b"[redacted]")
                log.write(line)
                log.flush()

        reader = threading.Thread(target=drain_log)
        reader.start()
        parent = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                memory = []
                for process in [parent, *parent.children(recursive=True)]:
                    try:
                        memory.append(process.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                ((utilization, memory_mib),) = _gpu()
                resources.write(
                    "resources",
                    gpu_pct=utilization,
                    gpu_mib=memory_mib,
                    rss_bytes=sum(memory),
                    parent_rss_bytes=memory[0] if memory else 0,
                )
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
            reader.join()
    (_output / "exit.json").write_text(
        json.dumps(
            {
                "exit": child.returncode,
                "prefetch": args.prefetch,
                "steps": args.steps,
            }
        )
        + "\n"
    )
    return child.returncode


def main():
    """Launch an idle-GPU run or execute its isolated training process."""
    global _output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefetch", required=True, choices=("off", "on"))
    parser.add_argument("--steps", type=int, default=24, choices=range(1, 25))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    _output = args.output.resolve()
    if args.child:
        try:
            _train(args.config, args.steps)
        except BaseException as exc:
            # SDK errors can contain a signed URL or authorization details.
            Events("failure").write("failure", error_type=type(exc).__name__)
            raise
        return 0
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
