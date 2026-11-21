"""Compare full H5 reads and window reads on a bounded list of local samples.

Use original OSS H5 objects copied to scratch. Counts are compressed HDF5
chunk payloads intersecting the selection, independent of metadata and cache.
"""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def _read(path, selections):
    start = perf_counter()
    with h5py.File(path, "r") as h5:
        values = {name: h5[name][selection] for name, selection in selections.items()}
    return perf_counter() - start, values


def _chunk_payload(dataset, selection):
    if dataset.chunks is None:
        return None
    coords = [
        range((axis.start // chunk) * chunk, axis.stop, chunk)
        for axis, chunk in zip(selection, dataset.chunks, strict=True)
    ]
    offsets = list(product(*coords))
    return len(offsets), sum(
        dataset.id.get_chunk_info_by_coord(offset).size for offset in offsets
    )


def _remote_reads(root, filename, modes, slices):
    results = {}
    for mode, selections in modes.items():
        with (root / filename).open("rb") as stream, h5py.File(stream, "r") as h5:
            arrays = {
                name: h5[name][selection] for name, selection in selections.items()
            }
            results[mode] = {
                "range_cache_misses": stream.cache.miss_count,
                "range_requested_bytes": stream.cache.total_requested_bytes,
                "arrays": arrays,
            }
    actual = results["slice"].pop("arrays")
    expected = results["full"].pop("arrays")
    for name, window in slices.items():
        np.testing.assert_array_equal(actual[name], expected[name][window])
    return results


def _inspect(path, modalities, repetitions, remote_root):
    slices, layouts = {}, {}
    with h5py.File(path, "r") as h5:
        for name in modalities:
            if name not in h5:
                continue
            ds = h5[name]
            if ds.ndim not in (3, 4):
                continue
            full = tuple(slice(0, size) for size in ds.shape)
            # A centered half-width crop and at most three stored time steps.
            # Use stored time indices, not logical timesteps after mask expansion.
            spatial = tuple(
                slice(size // 4, size // 4 + max(1, size // 2)) for size in ds.shape[:2]
            )
            tail = (
                (slice(0, min(ds.shape[2], 3)), full[-1]) if ds.ndim == 4 else full[-1:]
            )
            window = spatial + tail
            slices[name] = window
            props = ds.id.get_create_plist()
            layouts[name] = {
                "filter_ids": [
                    props.get_filter(i)[0] for i in range(props.get_nfilters())
                ],
                "shape": ds.shape,
                "chunks": ds.chunks,
                "full_chunks_and_bytes": _chunk_payload(ds, full),
                "crop_chunks_and_bytes": _chunk_payload(ds, window),
            }
    if not slices:
        raise ValueError(f"No selected datasets in {path.name}")
    modes = {"full": dict.fromkeys(slices, ()), "slice": slices}
    timings = defaultdict(list)
    for pair in range(repetitions):
        values = {}
        for name in ("full", "slice") if pair % 2 == 0 else ("slice", "full"):
            elapsed, values[name] = _read(path, modes[name])
            timings[name].append(elapsed)
        for name, selection in slices.items():
            np.testing.assert_array_equal(
                values["full"][name][selection], values["slice"][name]
            )
    return {
        "sample": path.name,
        "object_bytes": path.stat().st_size,
        "local_read_median_s": {name: median(times) for name, times in timings.items()},
        "modalities": layouts,
        "fsspec": (
            _remote_reads(remote_root, path.name, modes, slices)
            if remote_root
            else None
        ),
    }


def main():
    """Validate hyperslabs and report touched chunks and warm local timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--oss-root")
    parser.add_argument("--oss-endpoint")
    parser.add_argument("--oss-region")
    args = parser.parse_args()
    if not 1 <= len(args.files) <= 10 or not 1 <= args.repetitions <= 20:
        parser.error("Use 1–10 scratch files and 1–20 paired repetitions")
    remote_root = None
    if any((args.oss_root, args.oss_endpoint, args.oss_region)):
        if not all((args.oss_root, args.oss_endpoint, args.oss_region)):
            parser.error("Pass --oss-root, --oss-endpoint and --oss-region together")
        from upath import UPath

        from olmoearth_pretrain.data.dataset import OSSConfig

        config = OSSConfig(endpoint_url=args.oss_endpoint, region=args.oss_region)
        remote_root = UPath(args.oss_root, **config.storage_options())
    print(
        json.dumps(
            [
                _inspect(p, args.modalities, args.repetitions, remote_root)
                for p in args.files
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
