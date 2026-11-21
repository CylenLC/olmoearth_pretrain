"""Compare real local and s3fs HDF5 reads through the sample pipeline."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import h5py
import hdf5plugin
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data import dataset as dataset_module
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataloader import _OSSGetPrefetcher
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    _use_prefetched_oss_payload,
)
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py


@pytest.fixture
def h5_fixture(tmp_path):
    """Write a single-chunk H5 with compacted time, missing values and derived data."""
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(tmp_path),
        training_modalities=["sentinel2_l2a", "sentinel1", "worldcover_onehot", "ndvi"],
        dtype=np.float32,
    )
    path = dataset._get_h5_file_path(73)
    dataset.sample_indices = np.array([73])
    bands = len(Modality.SENTINEL2_L2A.band_order)
    data = np.arange(8 * 8 * 3 * bands, dtype=np.float32).reshape(8, 8, 3, bands)
    data[..., 0] = MISSING_VALUE
    mask = [False, True, False, True, True, False]
    with h5py.File(path.path, "w") as h5:
        h5["timestamps"] = np.arange(18).reshape(6, 3)
        h5.create_dataset(
            "sentinel2_l2a", data=data, chunks=data.shape, **hdf5plugin.Zstd(clevel=3)
        )
        h5["worldcover"] = np.full((8, 8, 1), 10, dtype=np.uint8)
        h5["srtm"] = np.zeros((8, 8, 1), dtype=np.float32)
        masks = h5.create_group(ConvertToH5py.missing_timesteps_mask_group_name)
        masks["sentinel2_l2a"] = mask
        masks["srtm"] = np.ones(len(mask), dtype=bool)
    return dataset, path


def test_local_and_fsspec_hdf5_sample_parity(h5_fixture, monkeypatch):
    """Seekable s3fs and its buffered object produce identical raw and cropped data."""
    pytest.importorskip("s3fs")
    local, local_path = h5_fixture
    payload = local_path.read_bytes()
    requests = []
    # The loopback endpoint stands in for Alibaba OSS. An incompatible
    # use_fsspec setting would try the boto3 path and bypass the server.
    monkeypatch.setattr(
        dataset_module, "_is_oss_full_get_path", lambda path: path.protocol == "s3"
    )

    class ObjectServer(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_HEAD(self):
            self.serve_object()

        def do_GET(self):
            self.serve_object()

        def serve_object(self):
            if self.path != f"/fixture/samples/{local_path.name}":
                self.send_error(404)
                return
            byte_range = self.headers.get("Range")
            requests.append((self.command, byte_range))
            data = payload
            if byte_range:
                first, last = map(int, byte_range.removeprefix("bytes=").split("-"))
                last = min(last, len(data) - 1)
                data = data[first : last + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {first}-{last}/{len(payload)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", '"fixture"')
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(data)

    with ThreadingHTTPServer(("127.0.0.1", 0), ObjectServer) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        rng_state = np.random.get_state()
        try:
            remote = copy.copy(local)
            remote.use_fsspec = True
            remote.h5py_dir = UPath(
                "s3://fixture/samples",
                anon=True,
                client_kwargs={
                    "endpoint_url": f"http://127.0.0.1:{server.server_port}"
                },
                config_kwargs={"s3": {"addressing_style": "path"}},
            )
            remote_path = remote._get_h5_file_path(73)
            expected_data, expected_masks = local.read_h5_file(local_path)
            remote_data, remote_masks = remote.read_h5_file(remote_path)
            assert (
                expected_data.keys()
                == remote_data.keys()
                == {"timestamps", "sentinel2_l2a", "worldcover_onehot"}
            )
            assert expected_masks.keys() == remote_masks.keys() == {"sentinel2_l2a"}
            for name in expected_data:
                np.testing.assert_array_equal(expected_data[name], remote_data[name])
            np.testing.assert_array_equal(
                expected_masks["sentinel2_l2a"], remote_masks["sentinel2_l2a"]
            )
            assert requests and any(
                method == "GET" and byte_range for method, byte_range in requests
            )

            for seed in (0, 1, 7):
                args = GetItemArgs(
                    idx=0, patch_size=2, sampled_hw_p=2, token_budget=256, target_t=2
                )
                np.random.seed(seed)
                patch, expected = local[args]
                assert expected.timestamps.shape == (2, 3)
                assert expected.sentinel2_l2a.shape[:3] == (4, 4, 2)
                assert np.all(expected.sentinel1 == MISSING_VALUE)
                assert np.all(expected.sentinel2_l2a[..., 0] == MISSING_VALUE)
                assert expected.ndvi.shape == (4, 4, 2, 1)
                np.random.seed(seed)
                remote_patch, actual = remote[args]
                np.random.seed(seed)
                before_buffered = len(requests)
                prefetcher = _OSSGetPrefetcher(2, remote_path.fs)
                try:
                    downloaded = prefetcher.submit(remote_path).result(timeout=5)
                finally:
                    prefetcher.close()
                after_download = len(requests)
                with _use_prefetched_oss_payload(remote_path, downloaded):
                    buffered_patch, buffered = remote[args]
                assert len(requests) == after_download > before_buffered
                assert remote_patch == buffered_patch == patch
                for candidate in (actual, buffered):
                    assert candidate.as_dict().keys() == expected.as_dict().keys()
                    for name, value in expected.as_dict().items():
                        np.testing.assert_array_equal(getattr(candidate, name), value)
        finally:
            np.random.set_state(rng_state)
            server.shutdown()
            thread.join()
