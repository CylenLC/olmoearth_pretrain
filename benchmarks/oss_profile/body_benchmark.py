"""Compare Full GET body readers with the same objects and W8/C2."""
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time

import psutil
from upath import UPath
from olmoearth_pretrain.data.dataset import OSSConfig, _get_oss_client

ROOT = "s3://olmoearth/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/3996"
IDS = (0, 570, 1141, 1712, 2282, 2853, 3424, 3995)

def worker(args):
    method, worker_id = args
    options = OSSConfig(endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com", region="cn-beijing").storage_options()
    paths = [UPath(f"{ROOT}/sample_{index}.h5", **options) for index in IDS] * 2
    before = resource.getrusage(resource.RUSAGE_SELF)
    def get(path):
        client = _get_oss_client(path)
        start = time.perf_counter()
        cpu_start = time.thread_time()
        response = client.get_object(Bucket=path.drive, Key=str(path).split("/", 3)[3])
        body = response["Body"]
        try:
            if method == "whole":
                payload = body.read()
            elif method == "chunks":
                pieces = []
                while True:
                    data = body.read(1024 * 1024)
                    if not data:
                        break
                    pieces.append(data)
                payload = b"".join(pieces)
            else:
                buf = bytearray(response["ContentLength"])
                pos = 0
                while pos < len(buf):
                    count = body.readinto(memoryview(buf)[pos:])
                    if not count:
                        break
                    pos += count
                body.readinto(bytearray(1))
                payload = bytes(buf) if pos == len(buf) else bytes(buf[:pos])
            if len(payload) != response["ContentLength"]:
                raise OSError("length mismatch")
            elapsed = time.perf_counter() - start
            cpu_seconds = time.thread_time() - cpu_start
            return {"id": path.name, "size": len(payload), "seconds": elapsed, "cpu_seconds": cpu_seconds,
                    "sha256": hashlib.sha256(payload).hexdigest()}
        finally:
            body.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(get, paths))
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {"worker": worker_id, "user": after.ru_utime-before.ru_utime,
            "system": after.ru_stime-before.ru_stime, "maxrss_kib": after.ru_maxrss,
            "rows": rows}

def main():
    output = []
    for method in ("whole", "chunks", "readinto", "readinto", "chunks", "whole"):
        net = psutil.net_io_counters()
        start = time.monotonic()
        with mp.get_context("spawn").Pool(8) as pool:
            rows = pool.map(worker, [(method, i) for i in range(8)])
        elapsed = time.monotonic()-start
        end = psutil.net_io_counters()
        output.append({"method": method, "wall": elapsed, "net_rx": end.bytes_recv-net.bytes_recv,
                       "net_tx": end.bytes_sent-net.bytes_sent, "workers": rows})
        with open(sys.argv[1], "w") as f:
            json.dump(output, f)
        print(method, round(elapsed, 2), flush=True)

if __name__ == "__main__":
    main()
