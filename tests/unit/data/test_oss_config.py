"""Offline coverage for native OSS UPath configuration and metadata reads."""

import json
import pickle
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig, OSSConfig


def oss_config() -> OSSConfig:
    """Return public, non-secret connection settings."""
    return OSSConfig(
        endpoint_url="https://s3.oss-cn-beijing.aliyuncs.com", region="cn-beijing"
    )


def test_upath_roundtrip_does_not_pickle_credentials(monkeypatch) -> None:
    """Spawn serialization keeps options and leaves credentials to the provider."""
    pytest.importorskip("s3fs")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-only-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-only-secret-key")
    config = OlmoEarthDatasetConfig(
        h5py_dir="s3://example/prefix/sentinel2_l2a/3",
        training_modalities=["sentinel2_l2a"],
        oss=oss_config(),
    )
    path = config.h5py_dir_upath
    # Initialize the filesystem in the parent, as Dataset.prepare() does.
    # This must still serialize only UPath options, never live connections.
    assert path.fs is path.fs
    serialized = pickle.dumps(path)
    restored = pickle.loads(serialized)
    assert restored.storage_options == path.storage_options
    child = restored / "sample_0.h5"
    assert child.storage_options == path.storage_options
    assert child.fs is path.fs
    assert child.storage_options["config_kwargs"]["signature_version"] == "s3"
    assert child.storage_options["config_kwargs"]["s3"]["addressing_style"] == "virtual"
    for secret in (b"test-only-access-key", b"test-only-secret-key"):
        assert secret not in serialized
        assert secret.decode() not in json.dumps(config.as_config_dict())
    assert {f.name for f in fields(OSSConfig)} == {"endpoint_url", "region", "profile"}


def test_config_serialization_and_cli() -> None:
    """Connection settings survive training config saves and CLI overrides."""
    pytest.importorskip("s3fs")
    config = OlmoEarthDatasetConfig(
        h5py_dir="s3://example/prefix/sentinel2_l2a/3",
        training_modalities=["sentinel2_l2a"],
    )
    config = config.merge(
        [
            "oss.endpoint_url=https://s3.oss-cn-beijing.aliyuncs.com",
            "oss.region=cn-beijing",
        ]
    )
    restored = OlmoEarthDatasetConfig.from_dict(config.as_config_dict())
    assert isinstance(restored.oss, OSSConfig)
    assert (
        restored.h5py_dir_upath.storage_options == config.h5py_dir_upath.storage_options
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com",
        "https://user:secret@example.com",
        "https://example.com/?token=secret",
        "https://example.com/#secret",
        "https://example.com/path",
        "https://",
    ],
)
def test_invalid_endpoint(endpoint: str) -> None:
    """Reject embedded credentials before paths or configs are logged."""
    with pytest.raises(ValueError, match="HTTPS origin"):
        OSSConfig(endpoint_url=endpoint, region="cn-beijing").storage_options()


def test_profile_and_local_path_validation(tmp_path) -> None:
    """Profiles use the standard provider and OSS options never affect local paths."""
    config = oss_config()
    config.profile = "training"
    assert config.storage_options()["profile"] == "training"
    config.region = ""
    with pytest.raises(ValueError, match="region"):
        config.storage_options()
    local = OlmoEarthDatasetConfig(str(tmp_path), ["sentinel2_l2a"])
    assert local.h5py_dir_upath == UPath(tmp_path)
    local.oss = oss_config()
    with pytest.raises(ValueError, match="s3://"):
        _ = local.h5py_dir_upath


def test_metadata_uses_configured_file_objects(setup_h5py_dir, monkeypatch) -> None:
    """Prepare reads CSV and NPY from UPath, preserving filesystem options."""
    import olmoearth_pretrain.data.dataset as module

    original = module.pd.read_csv

    def read_file_object(source, *args, **kwargs):
        assert hasattr(source, "read"), "CSV parsing must not reconstruct a URI"
        return original(source, *args, **kwargs)

    monkeypatch.setattr(module.pd, "read_csv", read_file_object)
    dataset = OlmoEarthDatasetConfig(str(setup_h5py_dir), ["sentinel2_l2a"]).build()
    dataset.prepare()
    np.testing.assert_array_equal(dataset.sample_indices, [0])
    assert dataset.latlon_distribution.shape == (1, 2)
    assert dataset.cache_dir is None


def test_shared_training_config_roundtrip(tmp_path) -> None:
    """The shared experiment builder routes OSS overrides into saved config.

    Construct real configs without importing optional downstream evaluation
    datasets. This checks configuration routing, not a full training launch.
    """
    pytest.importorskip("beaker")
    from olmo_core.optim import AdamWConfig
    from olmo_core.train.config import TrainerConfig

    from olmoearth_pretrain.config import Config
    from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
    from olmoearth_pretrain.internal.experiment import CommonComponents, build_config
    from olmoearth_pretrain.train.masking import MaskingConfig
    from olmoearth_pretrain.train.train_module.train_module import (
        OlmoEarthTrainModuleConfig,
    )

    fixture = next(
        p
        for p in (
            Path(__file__).resolve().parents[2] / "fixtures/checkpoint_configs"
        ).glob("*.json")
        if not p.name.endswith(".shapes.json")
    )
    model_config = Config.from_dict(json.loads(fixture.read_text())["model"])

    common = CommonComponents(
        run_name="oss-config-test",
        save_folder=str(tmp_path),
        training_modalities=["sentinel2_l2a", "sentinel1", "worldcover", "latlon"],
    )
    config = build_config(
        common=common,
        model_config_builder=lambda _: model_config,
        dataset_config_builder=lambda c: OlmoEarthDatasetConfig(
            str(tmp_path), c.training_modalities
        ),
        dataloader_config_builder=lambda _: OlmoEarthDataLoaderConfig(
            work_dir=str(tmp_path),
            global_batch_size=1,
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[8],
            seed=0,
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
        ),
        trainer_config_builder=lambda _: TrainerConfig(save_folder=str(tmp_path)),
        train_module_config_builder=lambda _: OlmoEarthTrainModuleConfig(
            optim_config=AdamWConfig(lr=1e-4), rank_microbatch_size=1
        ),
        overrides=[
            "--dataset.h5py_dir=s3://example/prefix/sentinel1_sentinel2_l2a/3",
            "--dataset.oss.endpoint_url=https://s3.oss-cn-beijing.aliyuncs.com",
            "--dataset.oss.region=cn-beijing",
        ],
    )
    restored = Config.from_dict(config.as_config_dict())
    assert isinstance(restored.dataset.oss, OSSConfig)
    assert restored.dataset.oss == oss_config()
    assert restored.data_loader.work_dir == str(tmp_path)
