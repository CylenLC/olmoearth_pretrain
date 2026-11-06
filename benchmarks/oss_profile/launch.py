"""Replay the historical command with a hard short-window cap."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def gpu_snapshot():
    device = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    apps_text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    gpus = []
    for line in device:
        index, utilization, memory = [item.strip() for item in line.split(",")]
        gpus.append({"index": int(index), "utilization_pct": int(utilization),
                     "memory_mib": int(memory)})
    apps = []
    for line in apps_text.splitlines():
        if line.strip():
            pid, memory = [item.strip() for item in line.split(",")]
            apps.append({"pid": int(pid), "memory_mib": int(memory)})
    return {"gpus": gpus, "compute_apps": apps}


def parse_cpu_list(value):
    result = set()
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def verify_clean_close(output, steps, worker_logs):
    """Verify early shutdown with actual GET activity and all worker exit codes."""
    parent = next(output.glob("parent-*.jsonl"))
    rows = [json.loads(line) for line in parent.read_text().splitlines()]
    closes = [row for row in rows if row["event"] == "close_end"]
    close = next(row for row in rows if row["event"] == "close_start")
    shutdown = next(row for row in rows if row["event"] == "shutdown_start")
    updates = [row for row in rows if row["event"] == "update"]
    requests = []
    starts = 0
    for log in worker_logs:
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row["event"] == "get_start":
                starts += 1
            elif row["event"] == "get_end":
                requests.append(row)
    workers = closes[0]["workers"]
    residual = [pid for pid in workers if psutil.pid_exists(int(pid))]
    stderr = (output / "train.log").read_text(errors="replace")
    in_flight = [row for row in requests
                 if row["start_ns"] <= close["ns"] < row["end_ns"]]
    result = {
        "steps": [row["step"] for row in updates],
        "close_after_last_update_s": (close["ns"] - updates[-1]["end_ns"]) / 1e9,
        "drain_s": (shutdown["ns"] - close["ns"]) / 1e9,
        "close_s": (closes[0]["ns"] - close["ns"]) / 1e9,
        "in_flight_at_close": len(in_flight),
        "in_flight_workers": sorted({row["worker"] for row in in_flight}),
        "get_starts": starts,
        "get_ends": len(requests),
        "get_errors": sum(row["error"] is not None for row in requests),
        "get_ends_after_native_shutdown": sum(row["end_ns"] > shutdown["ns"]
                                              for row in requests),
        "native_abort": any(token in stderr for token in (
            "Aborted", "SIGABRT", "terminate called without an active exception"
        )),
        "worker_exitcodes": workers,
        "worker_pids_remaining": residual,
    }
    (output / "close-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    if (result["steps"] != list(range(1, steps+1)) or not in_flight
            or starts != len(requests) or result["get_errors"]
            or result["get_ends_after_native_shutdown"] or result["native_abort"]
            or len(workers) != 8 or any(code != 0 for code in workers.values())
            or residual or len(closes) != 1):
        raise SystemExit("Early-close verification failed. See close-verification.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Historical boto3 command.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--first", type=int, default=9)
    parser.add_argument("--last", type=int, default=11)
    parser.add_argument("--gets-per-worker", type=int,
                        help="Override only the per-worker Full GET concurrency")
    parser.add_argument("--events-only", action="store_true",
                        help="Collect stage timings without activating Kineto")
    parser.add_argument("--verify-clean-close", action="store_true",
                        help="Require GETs in flight and eight clean worker exits")
    parser.add_argument("--parent-cpus", help="Restrict the training parent process, e.g. 0-3")
    parser.add_argument("--worker-cpus", help="Restrict DataLoader workers, e.g. 4-15")
    args = parser.parse_args()
    if not 1 <= args.first <= args.last <= args.steps <= 24:
        parser.error("Require 1 <= first <= last <= steps <= 24")
    if bool(args.parent_cpus) != bool(args.worker_cpus):
        parser.error("Pass both --parent-cpus and --worker-cpus for affinity isolation")
    if args.parent_cpus and parse_cpu_list(args.parent_cpus) & parse_cpu_list(args.worker_cpus):
        parser.error("Parent and worker CPU masks must be disjoint")
    script = Path(__file__).resolve().parent
    root = script.parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.error("Output must be inside the profiling worktree")
    baseline = json.loads(args.baseline.read_text())
    command = list(baseline["argv"])
    nano = next(i for i, value in enumerate(command) if value.endswith("/v1_2/nano.py"))
    command[nano] = str(script / "entry.py")
    for flag, value in {
        "--common.save_folder=": output / "results",
        "--trainer.max_duration.value=": args.steps,
    }.items():
        index = next(i for i, item in enumerate(command) if item.startswith(flag))
        command[index] = f"{flag}{value}"
    if "--trainer.max_duration.unit=steps" not in command:
        parser.error("The baseline duration must be in steps")
    if args.gets_per_worker is not None:
        if not 1 <= args.gets_per_worker <= 4:
            parser.error("--gets-per-worker must be between 1 and 4")
        option = "--data_loader.max_inflight_oss_gets_per_worker="
        index = next(i for i, item in enumerate(command) if item.startswith(option))
        command[index] = f"{option}{args.gets_per_worker}"

    preflight = gpu_snapshot()
    if any(gpu["utilization_pct"] > 0 or gpu["memory_mib"] > 0
           for gpu in preflight["gpus"]) or preflight["compute_apps"]:
        raise SystemExit(f"GPU is not idle; no training launched: {preflight}")

    profile = [] if args.events_only else [args.first, args.last]
    first, last = profile or (0, 0)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("OSS_PROFILE_PARENT_CPUS", None)
    env.pop("OSS_PROFILE_WORKER_CPUS", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root),
               OSS_PROFILE_OUTPUT=str(output), OSS_PROFILE_FIRST=str(first),
               OSS_PROFILE_LAST=str(last), PYTHONFAULTHANDLER="1",
               OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               CUDA_VISIBLE_DEVICES="0", WANDB_MODE="disabled", WANDB_SILENT="true")
    if args.parent_cpus:
        env["OSS_PROFILE_PARENT_CPUS"] = args.parent_cpus
        env["OSS_PROFILE_WORKER_CPUS"] = args.worker_cpus
    (output / "command.json").write_text(json.dumps({"argv": command, "cwd": str(root),
        "steps": args.steps, "profile_steps": profile,
        "cpu_affinity": {"parent": args.parent_cpus, "workers": args.worker_cpus}}, indent=2))
    (output / "gpu-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    start = time.monotonic()
    with (output / "train.log").open("wb", buffering=0) as log, (
        output / "resources.jsonl"
    ).open("w", buffering=1) as resources, (
        output / "gpu-samples.jsonl"
    ).open("w", buffering=1) as gpu_samples:
        proc = subprocess.Popen(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        psutil.cpu_times_percent(None)
        while proc.poll() is None:
            host = psutil.cpu_times_percent(None)
            try:
                parent = psutil.Process(proc.pid)
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            times = {}
            for process in processes:
                try:
                    ct = process.cpu_times()
                    times[process.pid] = [ct.user, ct.system]
                except psutil.NoSuchProcess:
                    pass
            gpu = gpu_snapshot()
            own_pids = set(times)
            gpu["foreign_compute_pids"] = [row["pid"] for row in gpu["compute_apps"]
                                           if row["pid"] not in own_pids]
            gpu_samples.write(json.dumps({"ns": time.monotonic_ns(), **gpu}) + "\n")
            resources.write(json.dumps({"ns": time.monotonic_ns(),
                "host_cpu_pct": 100-host.idle, "host_iowait_pct": host.iowait,
                "proc_times": times}) + "\n")
            time.sleep(1)
    result = {"exit": proc.wait(), "seconds": time.monotonic()-start}
    (output / "exit.json").write_text(json.dumps(result) + "\n")
    print(result, flush=True)
    if result["exit"]:
        return result["exit"]
    workers = sorted(output.glob("worker-*.jsonl"))
    if len(workers) != 8 or not all('"event":"get_end"' in f.read_text() for f in workers):
        raise SystemExit("Missing Full GET telemetry from one or more workers")
    if profile and not (output / "trace.json.gz").is_file():
        raise SystemExit("Missing CPU/CUDA trace")
    if args.verify_clean_close:
        verify_clean_close(output, args.steps, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
