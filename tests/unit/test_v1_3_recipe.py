"""Configuration round trips for the v1.3 training entrypoint."""

import json
from pathlib import Path

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.internal.experiment import CommonComponents, build_config
from scripts.official.v1_3 import base


def test_loss_config_can_be_rebuilt() -> None:
    """A second model/optimizer stack must retain the saved loss constructor."""
    common = CommonComponents("test", "local_output/test", [])
    loss_config = base.build_train_module_config(common).loss_config
    original = json.loads(json.dumps(loss_config.as_config_dict()))
    first = loss_config.build()
    second = loss_config.build()
    assert type(first) is type(second)
    assert loss_config.as_config_dict() == original


def test_checkpoint_model_provenance() -> None:
    """The training model must match the independently verified release fixture."""
    fixture = Path(__file__).parents[1] / "fixtures/checkpoint_configs"
    source = fixture / (
        "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1.json"
    )
    expected = Config.from_dict(json.loads(source.read_text())["model"])
    common = CommonComponents("test", "local_output/test", [])
    common.tokenization_config = expected.encoder_config.tokenization_config
    actual = base.build_model_config(common)
    assert actual.as_config_dict() == expected.as_config_dict()
    actual.validate()


def test_cli_override_roundtrip(tmp_path: Path) -> None:
    """Override output, data, batch and step count through the official config API."""
    model = Config.from_dict(json.loads(base.MODEL_CONFIG_PATH.read_text())["model"])
    common = CommonComponents(
        "test",
        "unused",
        model.encoder_config.supported_modality_names,
        tokenization_config=model.encoder_config.tokenization_config,
    )
    config = build_config(
        common=common,
        model_config_builder=base.build_model_config,
        dataset_config_builder=base.build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        train_module_config_builder=base.build_train_module_config,
        trainer_config_builder=base.build_trainer_config,
        overrides=[
            f"--common.save_folder={tmp_path}",
            "--dataset.h5py_dir=/data/3996",
            "--data_loader.global_batch_size=2",
            "--train_module.rank_microbatch_size=2",
            "--trainer.max_duration.value=3",
        ],
    )
    restored = Config.from_dict(json.loads(json.dumps(config.as_config_dict())))
    assert str(restored.dataset.h5py_dir) == "/data/3996"
    assert restored.data_loader.global_batch_size == 2
    assert restored.train_module.rank_microbatch_size == 2
    assert restored.trainer.max_duration.value == 3
    assert restored.trainer.save_folder == str(tmp_path)
    assert restored.trainer.work_dir == str(tmp_path)
    assert restored.data_loader.work_dir == str(tmp_path)
    assert restored.trainer.checkpointer.work_dir == str(tmp_path)
    restored.model.validate()
    restored.data_loader.validate()
