"""Summarize the original benchmark and the short profile from raw artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
from statistics import mean


def dump(data):
    print(json.dumps(data, indent=2))


def distribution(values):
    values = sorted(values)
    if not values:
        return None

    def quantile(q):
        pos = (len(values) - 1) * q
        index = int(pos)
        return values[index] + (pos-index) * (values[min(index+1, len(values)-1)] - values[index])

    return dict(n=len(values), mean=mean(values), p50=quantile(.5), p95=quantile(.95),
                p99=quantile(.99), max=values[-1])


def original(root):
    markers, data = {}, {}
    for line in (root / "train200.log").read_text().splitlines():
        match = re.search(r"\[step=(\d+)/200,epoch=1(?:,|\])", line)
        if match:
            step = int(match[1])
            if step not in markers:
                markers[step] = datetime.fromisoformat(line.split("\t", 1)[0]).timestamp()
                data[step] = {}
        if markers:
            metric = re.search(r"device/(data loading|model duration) \(s\)=([0-9.Ee+-]+)", line)
            if metric:
                data[step][metric[1]] = float(metric[2])
    begin, end = markers[8], markers[200]
    steps = [data[step] for step in range(9, 201)]
    tails = {step: data[step]["data loading"] for step in range(9, 201)
             if data[step]["data loading"] > 1}
    with (root / "system_samples.csv").open() as stream:
        sys = [r for r in csv.DictReader(stream) if begin <= datetime.fromisoformat(r["utc"]).timestamp() <= end]
    with (root / "process_memory.csv").open() as stream:
        process = defaultdict(list)
        for r in csv.DictReader(stream):
            if begin <= datetime.fromisoformat(r["utc"]).timestamp() <= end:
                process[r["pid"]].append(r)
    main_pid = min(process, key=int)
    cpu = defaultdict(lambda: [0, 0])
    for pid, rows in process.items():
        role = "worker" if rows[0]["name"] == "pt_data_worker" else "parent" if pid == main_pid else "other"
        for index, key in enumerate(("cpu_user_s", "cpu_system_s")):
            cpu[role][index] += (float(rows[-1][key])-float(rows[0][key])) / (end-begin)
    return dict(samples_per_s=192*512/(end-begin), duration=end-begin,
                tails=tails, tails_seconds=sum(tails.values()),
                load=distribution(s["data loading"] for s in steps),
                update=distribution(s["model duration"] for s in steps),
                host_cpu=mean(float(r["host_cpu_pct"]) for r in sys),
                gpu_busy=mean(float(r["gpu_util_pct"]) for r in sys if r["gpu_util_pct"]),
                user_system_cores=cpu, exit=int((root / "exit_status").read_text()))


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def interval_ms(row):
    return (row["end_ns"] - row["start_ns"]) / 1e6


def outer_ranges(rows, start, end):
    """Suppress nested copies of the same label on a single thread."""
    seen_end = {}
    for row in sorted(rows, key=lambda r: (r[start], -r[end])):
        thread = row["tid"]
        if row[end] > seen_end.get(thread, float("-inf")):
            seen_end[thread] = row[end]
            yield row


def short(root, first_step=9, last_step=None):
    events = defaultdict(list)
    for path in sorted([*root.glob("parent-*.jsonl"), *root.glob("worker-*.jsonl")]):
        for row in jsonl(path):
            events[row["event"]].append(row)
    command = json.loads((root / "command.json").read_text())
    if command["profile_steps"]:
        assert len(events["trace_saved"]) == 1, "No verified CPU/CUDA trace"
    assert sorted(r["worker"] for r in events["installed"] if r["role"] == "worker") == list(range(8))
    updates = sorted(events["update"], key=lambda r: r["step"])
    begin = next(r["start_ns"] for r in events["next"] if r["step"] == first_step)
    last_step = last_step or updates[-1]["step"]
    end = updates[last_step-1]["end_ns"]
    seconds = (end-begin)/1e9

    def in_window(event):
        rows = [r for r in events[event] if begin <= r["end_ns"] <= end]
        return list(outer_ranges(rows, "start_ns", "end_ns")) if event == "backward" else rows

    def cpu_s(rows, key="cpu_ns"):
        return sum(r[key] for r in rows)/1e9

    done = {(r["pid"], r["tid"], r["start_ns"]): r for r in events["get_end"]}
    missing = [r for r in events["get_start"] if (r["pid"], r["tid"], r["start_ns"]) not in done]
    active = []
    for r in events["get_end"]:
        start, stop = max(begin, r["start_ns"]), min(end, r["end_ns"])
        if stop > start:
            active.extend(((start, 1), (stop, -1)))
    for r in missing:
        start = max(begin, r["start_ns"])
        if start < end:
            active.extend(((start, 1), (end, -1)))
    peak = count = area = 0
    prev = begin
    for timestamp, change in sorted(active):
        area += (timestamp-prev)*count
        count += change
        peak = max(peak, count)
        prev = timestamp

    gets = in_window("get_end")
    successful = [r for r in gets if not r["error"] and "body_end_ns" in r]
    if gets:
        assert all(r["attempts"] >= 1 for r in gets), "GET HTTP observer coverage missing"
        assert all(r["expected_bytes"] == r["body_bytes"] == r["bytes"] for r in successful)
        assert len(successful) + sum(bool(r["error"]) for r in gets) == len(gets), "GET body observer coverage missing"
    get_metrics = dict(count=len(gets), bytes=sum(r["bytes"] or 0 for r in gets),
                       errors=Counter(r["error"] for r in gets if r["error"]),
                       statuses=Counter(s for r in gets for s in r.get("attempt_status", [])),
                       attempts=sum(r["attempts"] for r in gets),
                       all_run_incomplete=len(missing), peak=peak, mean=area/(end-begin),
                       total_ms=distribution(interval_ms(r) for r in successful),
                       headers_ms=distribution((r["headers_ns"]-r["request_ns"])/1e6 for r in successful),
                       body_ms=distribution((r["body_end_ns"]-r["body_start_ns"])/1e6 for r in successful),
                       cpu_ms=distribution(r["cpu_ns"]/1e6 for r in gets),
                       sdk_pre_send_post_cpu_s=(sum(r["headers_cpu_ns"]-r["request_cpu_ns"]-r["attempt_cpu_ns"]
                                                   for r in successful)/1e9),
                       attempt_cpu_s=cpu_s(gets, "attempt_cpu_ns"),
                       body_cpu_s=cpu_s(successful, "body_cpu_ns"))
    samples = in_window("sample")
    load_by_step = {r["step"]: r for r in events["loader_next"] if r["step"] is not None}
    ready_by_worker_batch = {(r["worker"], r["batch"]): r for r in events["batch_ready"]}
    waits = defaultdict(list)
    for row in events["ordered_wait"]:
        waits[row["worker"], row["batch"]].append(row)

    per_step = {}
    for row in events["next"]:
        step = row["step"]
        loader = load_by_step[step]
        key = ((step-1)%8, (step-1)//8)
        batch = ready_by_worker_batch[key]
        ordered = waits[key]
        overlap = sum(max(0, min(loader["end_ns"], r["end_ns"]) -
                          max(loader["start_ns"], r["start_ns"])) for r in ordered)/1e6
        per_step[step] = dict(loader_ms=interval_ms(loader), worker=key[0], batch=key[1],
                              ready_after_next_start_ms=(batch["end_ns"]-loader["start_ns"])/1e6,
                              loader_after_ready_ms=(loader["end_ns"]-batch["end_ns"])/1e6,
                              ordered_wait_overlap_ms=overlap,
                              worker_batch_ms=interval_ms(batch),
                              worker_batch_cpu_ms=batch["cpu_ns"]/1e6,
                              timestamps_shape=row["timestamps_shape"],
                              sentinel2_shape=row["sentinel2_shape"], patch=row["patch"],
                              update_ms=interval_ms(updates[step-1]))

    samples_host = [r for r in jsonl(root / "resources.jsonl") if begin <= r["ns"] <= end]
    roles = {str(r["pid"]): r["role"] for r in events["installed"]}
    start_resource, end_resource = samples_host[0], samples_host[-1]
    cores = defaultdict(lambda: [0, 0])
    for pid, last in end_resource["proc_times"].items():
        if pid in start_resource["proc_times"]:
            first = start_resource["proc_times"][pid]
            for index in range(2):
                cores[roles.get(pid, "other")][index] += (
                    (last[index]-first[index])*1e9 / (end_resource["ns"]-start_resource["ns"])
                )
    labels = ("h2d", "forward", "backward", "optimizer", "zero_grad", "pin")
    scheduler = sorted((r for r in events["main_sched"]
                       if first_step <= r["step"] <= last_step), key=lambda r: r["step"])
    shutdown_start = events["shutdown_start"][0]["ns"]
    completed = events["get_end"]
    ready = events["batch_ready"]
    shutdown_by_worker = {}
    for worker in range(8):
        worker_gets = [r for r in completed if r["worker"] == worker]
        worker_batches = [r for r in ready if r["worker"] == worker]
        shutdown_by_worker[worker] = {
            "last_get_after_shutdown_ms": (max(r["end_ns"] for r in worker_gets)-shutdown_start)/1e6,
            "last_ready_after_shutdown_ms": (max(r["end_ns"] for r in worker_batches)-shutdown_start)/1e6,
            "active_get_at_shutdown": sum(r["start_ns"] <= shutdown_start < r["end_ns"]
                                          for r in worker_gets),
        }
    final_update = updates[-1]["end_ns"]
    batch_size = next(int(item.split("=", 1)[1]) for item in command["argv"]
                      if item.startswith("--data_loader.global_batch_size="))
    gpu_samples = [json.loads(line) for line in (root / "gpu-samples.jsonl").read_text().splitlines()]
    window_gpu_samples = [r for r in gpu_samples if begin <= r["ns"] <= end]
    affinity = [r for r in events["affinity"] if r["role"] in ("parent", "worker")]
    affinity_masks = {str(r["worker"] if r["role"] == "worker" else "parent"):
                      sorted({cpu_id for mask in r["threads"].values() for cpu_id in mask})
                      for r in affinity}
    return dict(window_steps=[first_step, last_step], window_s=seconds,
                samples_per_s=(last_step-first_step+1)*batch_size/seconds,
                batch_size=batch_size,
                trace_saved_after_last_profile_step_s=(
                    (events["trace_saved"][0]["ns"] - updates[command["profile_steps"][-1]-1]["end_ns"])/1e9
                    if command["profile_steps"] else None
                ),
                all_run_get=dict(count=len(completed), bytes=sum(r["bytes"] or 0 for r in completed),
                                 attempts=sum(r["attempts"] for r in completed),
                                 errors=Counter(r["error"] for r in completed if r["error"])),
                shutdown_detail=dict(after_final_update_s=(shutdown_start-final_update)/1e9,
                                     batches_ready_before_final=sum(r["end_ns"] <= final_update for r in ready),
                                     batches_ready_total=len(ready), workers=shutdown_by_worker),
                get=get_metrics, sample_count=len(samples),
                sample_wall_ms=distribution(interval_ms(r) for r in samples),
                hdf5_read_ms=distribution(r["read_ns"]/1e6 for r in samples),
                hdf5_getitem_ms=distribution(r["h5_ns"]/1e6 for r in samples),
                ordered_wait_ms=distribution(r["wait_ns"]/1e6 for r in samples),
                worker_cpu_s=dict(get=cpu_s(gets), hdf5_read=cpu_s(samples, "read_cpu_ns"),
                                  hdf5_getitem=cpu_s(samples, "h5_cpu_ns"),
                                  assembly=cpu_s(samples)-cpu_s(samples, "read_cpu_ns"),
                                  collate=cpu_s(in_window("collate")),
                                  worker_fetch=cpu_s(in_window("batch_ready"))),
                phase_cpu_wall_ms={label: distribution(interval_ms(r) for r in in_window(label))
                                   for label in labels},
                main_thread_schedstat=dict(
                    samples=len(scheduler),
                    runqueue_wait_ms=distribution(r["runqueue_wait_ns"]/1e6 for r in scheduler),
                    runtime_ms=distribution(r["runtime_ns"]/1e6 for r in scheduler),
                    total_wait_s=sum(r["runqueue_wait_ns"] for r in scheduler)/1e9,
                    wait_fraction_of_step_wall=(
                        sum(r["runqueue_wait_ns"] for r in scheduler) /
                        sum(r["end_ns"]-r["start_ns"] for r in scheduler)
                        if scheduler else None
                    ),
                ),
                gpu_monitor=dict(
                    samples=len(window_gpu_samples),
                    mean_utilization_pct=mean(r["gpus"][0]["utilization_pct"]
                                              for r in window_gpu_samples),
                    max_utilization_pct=max(r["gpus"][0]["utilization_pct"]
                                            for r in window_gpu_samples),
                    mean_memory_mib=mean(r["gpus"][0]["memory_mib"]
                                         for r in window_gpu_samples),
                    foreign_compute_pids=sorted({pid for r in window_gpu_samples
                                                 for pid in r["foreign_compute_pids"]}),
                ),
                cpu_affinity_masks=affinity_masks,
                host_cpu=mean(r["host_cpu_pct"] for r in samples_host),
                user_system_cores=cores, steps=per_step,
                shutdown={label: events[label] for label in
                          ("shutdown_start", "shutdown_error", "shutdown_end")},
                exit=json.loads((root/"exit.json").read_text()))


def trace(root):
    with gzip.open(root / "trace.json.gz", "rt") as stream:
        events = json.load(stream)["traceEvents"]
    scopes = defaultdict(list)
    runtimes = {}
    gpu = []
    api = defaultdict(lambda: [0, 0.0])
    autograd_threads = set()
    for r in events:
        if r.get("ph") != "X":
            continue
        category = r.get("cat")
        if category == "user_annotation" and r["name"].startswith("oss_profile::"):
            scopes[r["tid"]].append(r)
        elif r["name"].startswith("autograd::engine::evaluate_function:"):
            autograd_threads.add(r["tid"])
        elif category in ("cuda_runtime", "cuda_driver"):
            corr = r.get("args", {}).get("correlation")
            if corr is not None:
                runtimes[corr] = r
            if "Synchronize" in r["name"]:
                api[r["name"]][0] += 1
                api[r["name"]][1] += r["dur"]/1e3
        elif category in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(r)
    for tid, rows in scopes.items():
        backward = [dict(r, stop=r["ts"]+r["dur"]) for r in rows
                    if r["name"] == "oss_profile::backward"]
        scopes[tid] = [r for r in rows if r["name"] != "oss_profile::backward"] + list(
            outer_ranges(backward, "ts", "stop")
        )
    backward_ranges = [r for rows in scopes.values() for r in rows
                       if r["name"] == "oss_profile::backward"]
    step_scopes = [r for rows in scopes.values() for r in rows
                   if r["name"] == "oss_profile::step"]
    summary = defaultdict(Counter)
    for rows in scopes.values():
        for r in rows:
            label = r["name"].split("::")[-1]
            summary[label]["calls"] += 1
            summary[label]["cpu_wall_ms"] += r["dur"]/1e3
    for r in gpu:
        caller = runtimes.get(r.get("args", {}).get("correlation"))
        label = "outside_named_phase" if caller else "uncorrelated"
        if caller:
            matches = [s for s in scopes[caller["tid"]]
                       if s["ts"] <= caller["ts"] <= s["ts"] + s["dur"]]
            if matches:
                label = min(matches, key=lambda s: s["dur"])["name"].split("::")[-1]
            elif caller["tid"] in autograd_threads and any(
                s["ts"] <= caller["ts"] <= s["ts"] + s["dur"] for s in backward_ranges
            ):
                label = "backward"
        kind = r.get("cat")
        summary[label][kind+"_ms"] += r["dur"]/1e3
        summary[label][kind+"_count"] += 1
        if kind == "gpu_memcpy":
            summary[label][r["name"].split(" (", 1)[0]+"_bytes"] += r.get("args", {}).get("bytes", 0)
    kernel_gaps = []
    per_step_gpu = []
    profile_steps = json.loads((root / "command.json").read_text()).get("profile_steps", [])
    first_trace_step = profile_steps[0] if profile_steps else None
    for offset, scope in enumerate(sorted(step_scopes, key=lambda r: r["ts"])):
        selected = []
        for activity in gpu:
            if activity.get("cat") != "kernel":
                continue
            caller = runtimes.get(activity.get("args", {}).get("correlation"))
            if caller and scope["ts"] <= caller["ts"] < scope["ts"] + scope["dur"]:
                selected.append((activity["ts"], activity["ts"]+activity["dur"]))
        selected.sort()
        gaps = []
        union = 0.0
        previous_end = None
        for start, end in selected:
            if previous_end is not None:
                if start > previous_end:
                    gaps.append(start-previous_end)
                if end > previous_end:
                    union += end-max(start, previous_end)
            else:
                union += end-start
            previous_end = max(previous_end or end, end)
        span = selected[-1][1]-selected[0][0] if selected else 0.0
        gap_total = max(0.0, span-union)
        kernel_gaps.extend(gaps)
        per_step_gpu.append(dict(
            step=first_trace_step+offset if first_trace_step is not None else None,
            kernel_count=len(selected),
            kernel_span_ms=span/1e3, kernel_active_union_ms=union/1e3,
            kernel_gap_ms=gap_total/1e3,
            inter_kernel_gap_ms=distribution(gap/1e3 for gap in gaps),
        ))
    # Union GPU activities across streams to avoid double-counting overlap.
    busy = 0.0
    previous_end = float("-inf")
    for r in sorted(gpu, key=lambda r: r["ts"]):
        end = r["ts"] + r["dur"]
        busy += max(0, end-max(r["ts"], previous_end))
        previous_end = max(previous_end, end)
    return dict(phases=summary, sync_api_calls_and_wall_ms=api,
                gpu_active_union_ms=busy/1e3, trace_gpu_count=len(gpu),
                per_step_cuda_kernel_gaps=per_step_gpu,
                cuda_inter_kernel_gap_ms=distribution(gap/1e3 for gap in kernel_gaps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["original", "short", "trace"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--first-step", type=int, default=9)
    parser.add_argument("--last-step", type=int)
    arguments = parser.parse_args()
    if arguments.kind == "short":
        dump(short(arguments.directory, arguments.first_step, arguments.last_step))
    else:
        dump({"original": original, "trace": trace}[arguments.kind](arguments.directory))
