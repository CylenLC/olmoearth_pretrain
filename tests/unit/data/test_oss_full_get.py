"""Regression coverage for complete OSS HDF5 reads."""

import io
import threading

import boto3
import h5py
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data import dataset as dataset_module
from olmoearth_pretrain.data import dataloader as dataloader_module
from olmoearth_pretrain.data.dataset import OlmoEarthDataset, OSSConfig
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py


def test_oss_full_get_matches_local_hdf5_and_reuses_client(tmp_path, monkeypatch):
    """The OSS path reads the same arrays, closes responses, and reuses its client."""
    local = tmp_path / "sample.h5"
    with h5py.File(local, "w") as f:
        f["timestamps"] = np.arange(6).reshape(3, 2)
        f["sentinel1"] = np.arange(24).reshape(2, 3, 4)
        f.create_group(ConvertToH5py.missing_timesteps_mask_group_name)["sentinel1"] = [
            True,
            False,
            True,
        ]
    payload = local.read_bytes()
    bodies = []
    calls = []

    class Client:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            body = io.BytesIO(payload)
            bodies.append(body)
            return {"Body": body, "ContentLength": len(payload)}

        def close(self):
            pass

    client = Client()
    sessions = []

    class Session:
        def __init__(self, profile_name=None):
            sessions.append(profile_name)

        def client(self, name, **kwargs):
            assert name == "s3"
            assert (
                kwargs["endpoint_url"] == "https://oss-cn-beijing-internal.aliyuncs.com"
            )
            return client

    monkeypatch.setattr(boto3, "Session", Session)
    monkeypatch.setattr(dataset_module._oss_clients, "signature", None, raising=False)
    monkeypatch.setattr(dataset_module._oss_clients, "client", None, raising=False)
    dataset = object.__new__(OlmoEarthDataset)
    dataset.cache_dir = None
    dataset.sec_per_sample = None
    dataset.training_modalities = ["sentinel1"]
    dataset.dtype = np.float32
    expected, expected_masks = dataset.read_h5_file(UPath(local))
    options = OSSConfig(
        endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com", region="cn-beijing"
    ).storage_options()
    path = UPath("s3://bucket/dir/sample.h5", **options)
    for _ in range(2):
        actual, masks = dataset.read_h5_file(path)
        assert actual.keys() == expected.keys()
        assert masks.keys() == expected_masks.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        for key in masks:
            np.testing.assert_array_equal(masks[key], expected_masks[key])
    assert sessions == [None]
    assert calls == [{"Bucket": "bucket", "Key": "dir/sample.h5"}] * 2
    assert all(body.closed for body in bodies)


def test_oss_full_get_closes_short_response(monkeypatch):
    """An incomplete object must not be handed to HDF5, and its body is closed."""
    body = io.BytesIO(b"short")

    class Client:
        def get_object(self, **kwargs):
            return {"Body": body, "ContentLength": 10}

        def close(self):
            pass

    class Session:
        def __init__(self, **kwargs):
            pass

        def client(self, *args, **kwargs):
            return Client()

    monkeypatch.setattr(boto3, "Session", Session)
    monkeypatch.setattr(dataset_module._oss_clients, "signature", None, raising=False)
    monkeypatch.setattr(dataset_module._oss_clients, "client", None, raising=False)
    options = OSSConfig(
        endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com", region="cn-beijing"
    ).storage_options()
    with (
        pytest.raises(OSError, match="length"),
        dataset_module._open_h5_file(UPath("s3://bucket/key", **options)),
    ):
        pass
    assert body.closed


def test_prefetched_oss_payload_uses_normal_hdf5_reader(monkeypatch, tmp_path):
    """Prefetched bytes enter the same Dataset parser without a second GET."""
    local = tmp_path / "sample.h5"
    with h5py.File(local, "w") as file:
        file["timestamps"] = np.arange(6).reshape(3, 2)
        file["sentinel1"] = np.arange(24).reshape(2, 3, 4)
    payload = local.read_bytes()
    path = UPath(
        "s3://bucket/sample.h5",
        **OSSConfig(
            endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com",
            region="cn-beijing",
        ).storage_options(),
    )

    def unexpected_session(*args, **kwargs):
        raise AssertionError("prefetched bytes should avoid a second OSS GET")

    monkeypatch.setattr(boto3, "Session", unexpected_session)
    dataset = object.__new__(OlmoEarthDataset)
    dataset.cache_dir = None
    dataset.sec_per_sample = None
    dataset.training_modalities = ["sentinel1"]
    dataset.dtype = np.float32

    with dataset_module._use_prefetched_oss_payload(path, payload):
        sample, masks = dataset.read_h5_file(path)
    np.testing.assert_array_equal(sample["timestamps"], np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(sample["sentinel1"], np.arange(24).reshape(2, 3, 4))
    assert masks == {}


def test_oss_get_prefetcher_bounds_parallelism_and_closes_clients(monkeypatch):
    """Per-worker full GET futures respect their configured bound."""
    active = 0
    peak = 0
    submitted = 0
    lock = threading.Lock()
    both_started = threading.Barrier(2, timeout=5)
    clients = []

    class Client:
        closed = False

        def close(self):
            self.closed = True

    def fake_download(path, on_client_create):
        nonlocal active, peak, submitted
        client = Client()
        on_client_create(client)
        with lock:
            submitted += 1
            sequence = submitted
            active += 1
            peak = max(peak, active)
        try:
            if sequence <= 2:
                both_started.wait()
            return path.name.encode()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(dataloader_module, "_download_oss_payload", fake_download)
    options = OSSConfig(
        endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com", region="cn-beijing"
    ).storage_options()
    paths = [UPath(f"s3://bucket/{idx}", **options) for idx in range(3)]
    prefetcher = dataloader_module._OSSGetPrefetcher(2)
    try:
        futures = [prefetcher.submit(path) for path in paths]
        assert [future.result(timeout=5) for future in futures] == [b"0", b"1", b"2"]
        peak = max(peak, active)
        clients = list(prefetcher._clients)
    finally:
        prefetcher.close()

    assert peak == 2
    assert clients and all(client.closed for client in clients)
