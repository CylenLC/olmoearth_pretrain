"""Opt-in observers for the v1.2 Nano OSS diagnostic, outside production code."""

import faulthandler
from functools import wraps
import json
import os
from pathlib import Path
import re
import threading
import time

clock = time.monotonic_ns
cpu = time.thread_time_ns
_state = threading.local()
_stream = None
_worker = None
_epoch = 0
_batch = 0
_step = None
_sample_pattern = re.compile(r"sample_(\d+)\.h5$")
_patches = set()
_request_seq = 0


def _cpu_list(value):
    cpus = set()
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            cpus.update(range(start, stop + 1))
        else:
            cpus.add(int(part))
    return cpus


def _thread_affinities():
    result = {}
    for item in Path("/proc/self/task").iterdir():
        tid = int(item.name)
        try:
            result[tid] = sorted(os.sched_getaffinity(tid))
        except ProcessLookupError:
            continue
    return result


def apply_affinity(role):
    """Apply and verify the opt-in parent/worker CPU split."""
    key = "OSS_PROFILE_PARENT_CPUS" if role == "parent" else "OSS_PROFILE_WORKER_CPUS"
    requested = os.environ.get(key)
    cpus = _cpu_list(requested) if requested else None
    if cpus is not None:
        tids = [int(item.name) for item in Path("/proc/self/task").iterdir()]
        for tid in tids:
            try:
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                continue
    observed = _thread_affinities()
    if cpus is not None and any(set(mask) != cpus for mask in observed.values()):
        raise RuntimeError(f"CPU affinity verification failed for {role}: {observed}")
    return {"role": role, "requested": sorted(cpus) if cpus is not None else None,
            "threads": observed}


def _schedstat(tid=None):
    tid = tid or threading.get_native_id()
    try:
        fields = Path(f"/proc/self/task/{tid}/schedstat").read_text().split()
        return tuple(map(int, fields[:3]))
    except (OSError, ValueError):
        return None


def emit(event, **fields):
    """Write allowlisted measurements and never serialize exception messages."""
    _stream.write(json.dumps({
        "event": event, "ns": clock(), "pid": os.getpid(),
        "tid": threading.get_native_id(), "worker": _worker,
        "task_id": getattr(_state, "task_id", None),
        "request_id": getattr(_state, "request_id", None), **fields,
    }, separators=(",", ":")) + "\n")


def open_log(role):
    global _stream
    if _stream is not None:
        return
    root = Path(os.environ["OSS_PROFILE_OUTPUT"])
    _stream = (root / f"{role}-{os.getpid()}.jsonl").open("x", buffering=1)
    emit("installed", role=role, wall_ns=time.time_ns())


def install_get():
    """Observe both the direct Dataset and aliased worker pool download paths."""
    if "get" in _patches:
        return
    from botocore.client import BaseClient
    from botocore.endpoint import Endpoint
    from botocore.response import StreamingBody
    from olmoearth_pretrain.data import dataloader, dataset

    original_download = dataset._download_oss_payload
    original_api = BaseClient._make_api_call
    original_send = Endpoint._send
    original_read = StreamingBody.read

    @wraps(original_download)
    def download(path, *args, **kwargs):
        match = _sample_pattern.search(path.name)
        start, start_cpu = clock(), cpu()
        ctx = {"attempts": 0, "attempt_ns": 0, "attempt_cpu_ns": 0}
        _state.get = ctx
        emit("get_start", start_ns=start, sample=int(match[1]) if match else None)
        payload = None
        error = None
        try:
            payload = original_download(path, *args, **kwargs)
            return payload
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end, end_cpu = clock(), cpu()
            del _state.get
            emit("get_end", start_ns=start, end_ns=end, cpu_ns=end_cpu-start_cpu,
                 error=error, bytes=len(payload) if payload is not None else None, **ctx)

    @wraps(original_api)
    def api(self, operation, params):
        ctx = getattr(_state, "get", None)
        if ctx is None or operation != "GetObject" or self.meta.service_model.service_name != "s3":
            return original_api(self, operation, params)
        ctx["request_ns"], ctx["request_cpu_ns"] = clock(), cpu()
        response = original_api(self, operation, params)
        ctx["headers_ns"], ctx["headers_cpu_ns"] = clock(), cpu()
        ctx["status"] = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        ctx["sdk_retries"] = response.get("ResponseMetadata", {}).get("RetryAttempts")
        ctx["expected_bytes"] = response.get("ContentLength")
        return response

    @wraps(original_send)
    def send(self, request):
        ctx = getattr(_state, "get", None)
        if ctx is None or "request_ns" not in ctx:
            return original_send(self, request)
        start, start_cpu = clock(), cpu()
        ctx["attempts"] += 1
        try:
            response = original_send(self, request)
            ctx.setdefault("attempt_status", []).append(response.status_code)
            return response
        except BaseException as exc:
            ctx.setdefault("attempt_errors", []).append(type(exc).__name__)
            raise
        finally:
            ctx["attempt_ns"] += clock() - start
            ctx["attempt_cpu_ns"] += cpu() - start_cpu

    @wraps(original_read)
    def read(self, *args, **kwargs):
        ctx = getattr(_state, "get", None)
        if ctx is None:
            return original_read(self, *args, **kwargs)
        ctx["body_start_ns"], start_cpu = clock(), cpu()
        try:
            data = original_read(self, *args, **kwargs)
            ctx["body_bytes"] = len(data)
            return data
        finally:
            ctx["body_end_ns"] = clock()
            ctx["body_cpu_ns"] = cpu() - start_cpu

    dataset._download_oss_payload = dataloader._download_oss_payload = download
    BaseClient._make_api_call = api
    Endpoint._send = send
    StreamingBody.read = read
    _patches.add("get")


def worker_init(worker_id):
    """Run the existing initializer and explicitly install in every spawn worker."""
    affinity = apply_affinity("worker")
    import h5py
    import torch
    from torch.utils.data._utils.fetch import _IterableDatasetFetcher
    from multiprocessing.queues import Queue
    from concurrent.futures import ThreadPoolExecutor

    from olmoearth_pretrain.data import dataloader
    from olmoearth_pretrain.data.dataloader import _OSSGetPrefetcher, OlmoEarthDataLoader
    from olmoearth_pretrain.data.dataset import OlmoEarthDataset

    global _worker
    if dataloader._worker_ignore_sigterm is worker_init:
        _original_worker_init(worker_id)
    else:
        dataloader._worker_ignore_sigterm(worker_id)
    _worker = worker_id
    open_log("worker")
    emit("affinity", **affinity)
    faulthandler.enable(all_threads=True)
    install_get()
    original_queue_get, original_queue_put = Queue.get, Queue.put
    worker_tid = threading.get_native_id()

    def queue_get(self, *args, **kwargs):
        result = original_queue_get(self, *args, **kwargs)
        if (threading.get_native_id() == worker_tid and isinstance(result, tuple)
                and len(result) == 2 and isinstance(result[0], int)):
            _state.task_id = result[0]
            emit("worker_task_received")
        return result

    def queue_put(self, value, *args, **kwargs):
        if (threading.get_native_id() == worker_tid and isinstance(value, tuple)
                and len(value) == 2 and isinstance(value[0], int)):
            start = clock()
            try:
                return original_queue_put(self, value, *args, **kwargs)
            finally:
                emit("worker_result_queued", task_id=value[0], start_ns=start, end_ns=clock())
        return original_queue_put(self, value, *args, **kwargs)

    Queue.get, Queue.put = queue_get, queue_put
    original_thread_submit = ThreadPoolExecutor.submit

    def thread_submit(self, fn, *args, **kwargs):
        if getattr(self, "_thread_name_prefix", "") != "olmoearth-oss-get":
            return original_thread_submit(self, fn, *args, **kwargs)
        task_id = getattr(_state, "task_id", None)
        request_id = getattr(_state, "request_id", None)

        def run():
            _state.task_id, _state.request_id = task_id, request_id
            try:
                return fn(*args, **kwargs)
            finally:
                _state.task_id = _state.request_id = None

        return original_thread_submit(self, run)

    ThreadPoolExecutor.submit = thread_submit
    dataset = torch.utils.data.get_worker_info().dataset
    loader = dataset.data_loader
    original_collate = loader.collator
    original_submit = _OSSGetPrefetcher.submit
    original_item = OlmoEarthDataLoader._get_dataset_item
    original_read = OlmoEarthDataset.read_h5_file
    original_h5 = h5py.Dataset.__getitem__
    original_fetch = _IterableDatasetFetcher.fetch

    def fetch(self, *args, **kwargs):
        global _batch
        start, start_cpu = clock(), cpu()
        try:
            result = original_fetch(self, *args, **kwargs)
        except BaseException as exc:
            emit("fetch_error", batch=_batch, error=type(exc).__name__)
            raise
        emit("batch_ready", batch=_batch, start_ns=start, end_ns=clock(),
             cpu_ns=cpu()-start_cpu, epoch=_epoch)
        _batch += 1
        return result

    def submit(self, path):
        global _request_seq
        _request_seq += 1
        request_id = _request_seq
        _state.request_id = request_id
        try:
            future = original_submit(self, path)
        finally:
            _state.request_id = None
        emit("future_submitted", request_id=request_id)
        original_result = future.result
        match = _sample_pattern.search(path.name)
        sample = int(match[1]) if match else None

        def result(*args, **kwargs):
            start = clock()
            try:
                return original_result(*args, **kwargs)
            finally:
                _state.wait_ns = clock() - start
                emit("future_result", request_id=request_id, start_ns=start,
                     end_ns=start+_state.wait_ns)
                if _state.wait_ns >= 10_000_000:
                    emit("ordered_wait", batch=_batch, sample=sample,
                         start_ns=start, end_ns=start+_state.wait_ns)

        future.result = result
        return future

    def item(self, index, *args, **kwargs):
        start, start_cpu = clock(), cpu()
        _state.read_ns = _state.read_cpu_ns = _state.h5_ns = _state.h5_cpu_ns = 0
        try:
            return original_item(self, index, *args, **kwargs)
        finally:
            emit("sample", batch=_batch, index=int(index), start_ns=start,
                 end_ns=clock(), cpu_ns=cpu()-start_cpu,
                 read_ns=_state.read_ns, read_cpu_ns=_state.read_cpu_ns,
                 h5_ns=_state.h5_ns, h5_cpu_ns=_state.h5_cpu_ns,
                 wait_ns=getattr(_state, "wait_ns", 0))
            _state.wait_ns = 0

    def read(self, path):
        start, start_cpu = clock(), cpu()
        try:
            return original_read(self, path)
        finally:
            _state.read_ns = clock() - start
            _state.read_cpu_ns = cpu() - start_cpu

    def h5(self, key):
        start, start_cpu = clock(), cpu()
        try:
            return original_h5(self, key)
        finally:
            if hasattr(_state, "h5_ns"):
                _state.h5_ns += clock() - start
                _state.h5_cpu_ns += cpu() - start_cpu

    def collate(batch):
        start, start_cpu = clock(), cpu()
        try:
            return original_collate(batch)
        finally:
            emit("collate", batch=_batch, start_ns=start, end_ns=clock(),
                 cpu_ns=cpu()-start_cpu)

    original_epoch = dataset._load_epoch_state

    def load_epoch():
        global _epoch
        original_epoch()
        _epoch = loader.epoch
        emit("epoch", epoch=_epoch)

    dataset._load_epoch_state = load_epoch
    loader.collator = collate
    _OSSGetPrefetcher.submit = submit
    OlmoEarthDataLoader._get_dataset_item = item
    OlmoEarthDataset.read_h5_file = read
    h5py.Dataset.__getitem__ = h5
    _IterableDatasetFetcher.fetch = fetch


def install_training(first, last, affinity):
    """Attach the profiler to real steps only. No synchronization inside phases."""
    import torch
    from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
    from torch.utils.data._utils import pin_memory

    from olmoearth_pretrain.data import dataloader
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    from olmoearth_pretrain.train.callbacks.speed_monitor import OlmoEarthSpeedMonitorCallback
    from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
        ContrastiveLatentMIMTrainModule as TrainModule,
    )

    global _original_worker_init
    open_log("parent")
    actual_affinity = _thread_affinities()
    expected = affinity["requested"]
    if expected is not None and any(mask != expected for mask in actual_affinity.values()):
        raise RuntimeError(f"Parent affinity changed after torch import: {actual_affinity}")
    emit("affinity", role="parent", requested=expected, threads=actual_affinity)
    _original_worker_init = dataloader._worker_ignore_sigterm
    dataloader._worker_ignore_sigterm = worker_init
    install_get()
    original_pin_loop = pin_memory._pin_memory_loop

    class QueueObserver:
        def __init__(self, queue, direction):
            self.queue, self.direction = queue, direction

        def __getattr__(self, name):
            return getattr(self.queue, name)

        def get(self, *args, **kwargs):
            result = self.queue.get(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], int):
                _state.task_id = result[0]
                emit("ipc_received")
            return result

        def put(self, value, *args, **kwargs):
            result = self.queue.put(value, *args, **kwargs)
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
                emit("pin_output", task_id=value[0])
                _state.task_id = None
            return result

    def pin_loop(in_queue, out_queue, *args):
        return original_pin_loop(QueueObserver(in_queue, "in"),
                                 QueueObserver(out_queue, "out"), *args)

    pin_memory._pin_memory_loop = pin_loop
    original_dispatch = _MultiProcessingDataLoaderIter._try_put_index
    original_get_data = _MultiProcessingDataLoaderIter._get_data
    original_process_data = _MultiProcessingDataLoaderIter._process_data

    def dispatch(self):
        task_id = self._send_idx
        result = original_dispatch(self)
        if self._send_idx != task_id:
            emit("task_dispatched", task_id=task_id, target_worker=self._task_info[task_id][0])
        return result

    def get_data(self):
        start = clock()
        result = original_get_data(self)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], int):
            emit("parent_queue_received", task_id=result[0], expected_task_id=self._rcvd_idx,
                 start_ns=start, end_ns=clock())
        return result

    def process_data(self, data, worker_idx):
        task_id = self._rcvd_idx - 1
        result = original_process_data(self, data, worker_idx)
        emit("consumer_delivered", task_id=task_id, target_worker=worker_idx, step=_step)
        return result

    _MultiProcessingDataLoaderIter._try_put_index = dispatch
    _MultiProcessingDataLoaderIter._get_data = get_data
    _MultiProcessingDataLoaderIter._process_data = process_data
    root = Path(os.environ["OSS_PROFILE_OUTPUT"])
    state = {}
    original_pre_load = OlmoEarthSpeedMonitorCallback.pre_load_batch
    original_pre = OlmoEarthSpeedMonitorCallback.pre_step
    original_post = OlmoEarthSpeedMonitorCallback.post_step

    def pre_load(self):
        global _step
        _step = self.trainer.global_step + 1
        if _step == first:
            state["profiler"] = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
            ])
            state["profiler"].start()
        state["load_start"] = clock()
        state["sched_tid"] = threading.get_native_id()
        state["sched_start"] = _schedstat(state["sched_tid"])
        if first <= _step <= last:
            state["step_scope"] = torch.profiler.record_function("oss_profile::step")
            state["step_scope"].__enter__()
        return original_pre_load(self)

    def pre(self, batch):
        end = clock()
        view = batch[1]
        emit("next", step=_step, start_ns=state["load_start"], end_ns=end,
             patch=int(batch[0]), timestamps_shape=list(view.timestamps.shape),
             sentinel2_shape=list(view.sentinel2_l2a.shape)
             if getattr(view, "sentinel2_l2a", None) is not None else None)
        state["model_start"] = clock()
        return original_pre(self, batch)

    def post(self):
        global _step
        end = clock()
        sched_end = _schedstat(state["sched_tid"])
        sched_start = state.pop("sched_start", None)
        if sched_start is not None and sched_end is not None:
            emit("main_sched", step=_step, tid=state["sched_tid"],
                 start_ns=state["load_start"], end_ns=end,
                 runtime_ns=sched_end[0]-sched_start[0],
                 runqueue_wait_ns=sched_end[1]-sched_start[1],
                 timeslices=sched_end[2]-sched_start[2])
        scope = state.pop("step_scope", None)
        if scope is not None:
            scope.__exit__(None, None, None)
        emit("update", step=_step, start_ns=state["model_start"], end_ns=end)
        result = original_post(self)
        if _step == last:
            state["profiler"].stop()
            emit("trace_stopped", first=first, last=last)
        _step = None
        return result

    def wrap(cls, name, label, always=False):
        original = getattr(cls, name)

        @wraps(original)
        def observed(self, *args, **kwargs):
            step = _step
            if step is None and not always:
                return original(self, *args, **kwargs)
            # Tensor subclasses can redispatch Tensor.backward on the same thread.
            if label == "backward" and getattr(_state, "backward", False):
                return original(self, *args, **kwargs)
            if label == "backward":
                _state.backward = True
            start, start_cpu = clock(), cpu()
            try:
                if step is not None and first <= step <= last:
                    with torch.profiler.record_function("oss_profile::" + label):
                        return original(self, *args, **kwargs)
                return original(self, *args, **kwargs)
            finally:
                if label == "backward":
                    _state.backward = False
                emit(label, step=step, start_ns=start, end_ns=clock(),
                     cpu_ns=cpu()-start_cpu)

        setattr(cls, name, observed)

    OlmoEarthSpeedMonitorCallback.pre_load_batch = pre_load
    OlmoEarthSpeedMonitorCallback.pre_step = pre
    OlmoEarthSpeedMonitorCallback.post_step = post
    wrap(MaskedOlmoEarthSample, "to_device", "h2d")
    wrap(TrainModule, "model_forward", "forward")
    wrap(torch.Tensor, "backward", "backward")
    wrap(TrainModule, "optim_step", "optimizer")
    wrap(TrainModule, "zero_grads", "zero_grad")
    wrap(_MultiProcessingDataLoaderIter, "__next__", "loader_next", always=True)
    wrap(dataloader._MaskedViewBatch, "pin_memory", "pin", always=True)

    shutdown = _MultiProcessingDataLoaderIter._shutdown_workers

    def observed_shutdown(self):
        if getattr(self, "_shutdown", True):
            return shutdown(self)
        workers = getattr(self, "_workers", ())
        emit("shutdown_start", workers=[p.pid for p in workers])
        try:
            return shutdown(self)
        except BaseException as exc:
            emit("shutdown_error", error=type(exc).__name__)
            raise
        finally:
            emit("shutdown_end", workers={p.pid: p.exitcode for p in workers})

    _MultiProcessingDataLoaderIter._shutdown_workers = observed_shutdown

    original_close = dataloader.OlmoEarthDataLoader.close

    def observed_close(self):
        iterator = getattr(self._torch_loader, "_iterator", None)
        if iterator is None:
            return original_close(self)
        workers = iterator._workers
        emit("close_start", workers=[p.pid for p in workers])
        try:
            return original_close(self)
        finally:
            emit("close_end", workers={p.pid: p.exitcode for p in workers},
                 alive=[p.pid for p in workers if p.is_alive()])

    dataloader.OlmoEarthDataLoader.close = observed_close

    def export():
        profiler = state.pop("profiler", None)
        if profiler is not None:
            profiler.export_chrome_trace(str(root / "trace.json.gz"))
            emit("trace_saved", first=first, last=last)

    return export
