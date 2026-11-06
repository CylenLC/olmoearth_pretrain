"""Join opt-in DataLoader observer events by PyTorch task and request ID."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    tasks = defaultdict(lambda: defaultdict(list))
    requests = defaultdict(dict)
    loaders = {}
    for path in sorted(args.run.glob("parent-*.jsonl")) + sorted(args.run.glob("worker-*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                event, task = row["event"], row.get("task_id")
                if event == "loader_next" and row.get("step") is not None:
                    loaders[row["step"]] = row
                if task is not None:
                    tasks[task][event].append(row)
                if event in ("get_start", "get_end") and row.get("request_id") is not None:
                    requests[row["worker"], row["request_id"]][event] = row

    def first(events, name):
        rows = events.get(name, [])
        return min(rows, key=lambda r: r["ns"]) if rows else None

    def duration(row):
        return (row["end_ns"] - row["start_ns"]) / 1e6 if row else None

    def delta(end, start):
        return (end - start) / 1e6 if end is not None and start is not None else None

    output = []
    pins = [row for grouped in tasks.values() for row in grouped.get("pin", [])]
    for task_id, events in sorted(tasks.items()):
        delivered = first(events, "consumer_delivered")
        if not delivered or delivered.get("step") is None:
            continue
        step = delivered["step"]
        loader = loaders.get(step)
        dispatched = first(events, "task_dispatched")
        received = first(events, "worker_task_received")
        ready = first(events, "batch_ready")
        queued = first(events, "worker_result_queued")
        ipc = first(events, "ipc_received")
        pin_out = first(events, "pin_output")
        parent = first(events, "parent_queue_received")
        pin = first(events, "pin")
        collate = first(events, "collate")
        future = events.get("future_result", [])
        samples = events.get("sample", [])
        joined_gets = [requests.get((r["worker"], r["request_id"]), {}).get("get_end")
                       for r in future]
        joined_gets = [r for r in joined_gets if r is not None]
        result = dict(
            task_id=task_id, step=step, worker=delivered["target_worker"],
            next_ms=duration(loader),
            dispatch_to_worker_ms=delta(received["ns"] if received else None,
                                        dispatched["ns"] if dispatched else None),
            worker_fetch_ms=duration(ready), collate_ms=duration(collate),
            future_wait_total_ms=sum(duration(r) for r in future),
            future_wait_max_ms=max((duration(r) for r in future), default=None),
            sample_count=len(samples),
            hdf5_read_cpu_ms=sum(r["read_cpu_ns"] for r in samples)/1e6,
            hdf5_getitem_cpu_ms=sum(r["h5_cpu_ns"] for r in samples)/1e6,
            futures=len(future), matched_gets=len(joined_gets),
            get_ms_max=max((duration(r) for r in joined_gets), default=None),
            queue_put_ms=duration(queued),
            queued_to_ipc_received_ms=delta(ipc["ns"] if ipc else None,
                                             queued["end_ns"] if queued else None),
            ipc_to_pin_output_ms=delta(pin_out["ns"] if pin_out else None,
                                        ipc["ns"] if ipc else None),
            pin_ms=duration(pin),
            pin_output_to_parent_ms=delta(parent["ns"] if parent else None,
                                           pin_out["ns"] if pin_out else None),
            parent_to_delivered_ms=delta(delivered["ns"], parent["ns"] if parent else None),
            expected_when_received=parent.get("expected_task_id") if parent else None,
            missing=[name for name in ("task_dispatched", "worker_task_received", "batch_ready",
                                       "worker_result_queued", "ipc_received", "pin_output",
                                       "parent_queue_received", "consumer_delivered")
                     if not events.get(name)],
        )
        if loader and ready:
            result["worker_ready_after_next_start_ms"] = delta(ready["end_ns"], loader["start_ns"])
            result["next_after_worker_ready_ms"] = delta(loader["end_ns"], ready["end_ns"])
        if loader:
            begin, end = loader["start_ns"], loader["end_ns"]
            result["pins_overlapping_next"] = [
                {"task_id": row["task_id"], "pin_ms": duration(row),
                 "overlap_ms": (min(end, row["end_ns"]) -
                                max(begin, row["start_ns"])) / 1e6}
                for row in pins if row["start_ns"] < end and row["end_ns"] > begin
            ]
            result["ipc_after_next_start_ms"] = delta(ipc["ns"] if ipc else None, begin)
            result["pin_output_after_next_start_ms"] = delta(pin_out["ns"] if pin_out else None, begin)
        output.append(result)
    print(json.dumps({"tasks": output}, indent=2))


if __name__ == "__main__":
    main()
