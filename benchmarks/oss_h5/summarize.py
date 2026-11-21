"""Summarize Nano OSS measurements from one or more completed 24-step runs."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _events(directory, pattern):
    for file in sorted(directory.glob(pattern)):
        with file.open() as stream:
            yield from map(json.loads, stream)


def _percentiles(values):
    sorted_values = sorted(values)
    if not sorted_values:
        return None
    result = {}
    for percentile in (50, 95):
        rank = (len(sorted_values) - 1) * percentile / 100
        lower, upper = math.floor(rank), math.ceil(rank)
        result[f"p{percentile}"] = sorted_values[lower] + (rank - lower) * (
            sorted_values[upper] - sorted_values[lower]
        )
    return result


def _io(events):
    requests = [row for row in events if row["event"] == "http"]
    return {
        "requests": dict(Counter(row["method"] for row in requests)),
        "request_status": dict(Counter(str(row["status"]) for row in requests)),
        "body_bytes": sum(row["bytes"] for row in events if row["event"] == "body"),
    }


def _summarize(path, warm_step):
    result = json.loads((path / "exit.json").read_text())
    if result["exit"]:
        raise ValueError(f"Run failed: {path.name}")
    trainer = list(_events(path, "trainer-*.jsonl"))
    steps = {
        event: {row["step"]: row for row in trainer if row["event"] == event}
        for event in ("load_start", "batch", "step")
    }
    if not 1 <= warm_step <= result["steps"]:
        raise ValueError("The warm window must start within the measured run")
    expected = set(range(1, result["steps"] + 1))
    if set(steps["step"]) != expected or set(steps["batch"]) != expected:
        raise ValueError(f"Incomplete training timing: {path.name}")
    waits = [
        steps["batch"][step]["time"] - steps["load_start"][step]["time"]
        for step in sorted(expected)
    ]
    end = steps["step"][result["steps"]]["time"]
    start = steps["load_start"][1]["time"]
    warm_start = steps["load_start"][warm_step]["time"]
    n_seen = sum(
        steps["batch"][step]["samples"] for step in expected if step >= warm_step
    )
    resources = [
        row for row in _events(path, "resources-*.jsonl") if start <= row["time"] <= end
    ]
    warm_resources = [row for row in resources if row["time"] >= warm_start]
    workers = list(_events(path, "worker-*.jsonl"))
    samples = [
        row for row in workers if row["event"] == "sample" and row["time"] <= end
    ]
    download = [
        row for row in workers if row["event"] == "download" and row["time"] <= end
    ]
    worker_batches = defaultdict(list)
    for row in workers:
        if row["event"] == "batch_indices":
            worker_batches[row["worker"]].append(row["digest"])
    # This loader emits batches in round-robin order across workers.
    assigned = [
        worker_batches[batch % len(worker_batches)][batch // len(worker_batches)]
        for batch in range(result["steps"])
    ]
    content_hash = hashlib.sha256("".join(assigned).encode()).hexdigest()
    return {
        **result,
        "first_batch_s": waits[0],
        "warm_samples_per_s": n_seen / (end - warm_start),
        "warm_step_samples_per_s": _percentiles(
            steps["batch"][step]["samples"]
            / (steps["step"][step]["time"] - steps["load_start"][step]["time"])
            for step in expected
            if step >= warm_step
        ),
        "warm_batch_wait_s": _percentiles(waits[warm_step - 1 :]),
        "max_warm_wait_s": max(waits[warm_step - 1 :]),
        "batch_waits_s": waits,
        "warm_wait_over_one_second": [
            step
            for step, wait in enumerate(waits, 1)
            if step >= warm_step and wait > 1.0
        ],
        "warm_model_s": _percentiles(
            steps["step"][step]["time"] - steps["batch"][step]["time"]
            for step in expected
            if step >= warm_step
        ),
        "out_of_order_arrivals": sum(row["event"] == "out_of_order" for row in trainer),
        "gpu_util_pct": mean(row["gpu_pct"] for row in warm_resources),
        "gpu_peak_mib": max(row["gpu_mib"] for row in resources),
        "rss_summed_peak_bytes": max(row["rss_bytes"] for row in resources),
        "worker_io_through_last_step": _io(
            [row for row in workers if row["time"] <= end]
        ),
        "worker_io_total": _io(workers),
        "setup_io": _io(list(_events(path, "setup-*.jsonl"))),
        "samples_built_through_last_step": len(samples),
        "sample_seconds": {
            key: _percentiles(row[key] for row in samples)
            for key in (
                "open_s",
                "range_s",
                "read_h5_s",
                "h5_read_s",
                "h5_open_s",
                "construct_s",
                "crop_s",
                "crop_normalize_s",
                "future_wait_s",
            )
        },
        "object_open_s": _percentiles(row["open_s"] for row in (download or samples)),
        "object_read_s": _percentiles(
            (row["transfer_s"] for row in download)
            if download
            else (row["range_s"] for row in samples)
        ),
        "object_open_read_s": _percentiles(
            (row["seconds"] for row in download)
            if download
            else (row["open_s"] + row["range_s"] for row in samples)
        ),
        "index_order_sha256": content_hash,
    }


def main():
    """Print the same timing, resource and request summary for each run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--warm-step", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            {run.name: _summarize(run, args.warm_step) for run in args.runs}, indent=2
        )
    )


if __name__ == "__main__":
    main()
