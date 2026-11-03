"""OlmoEarth v1.3 Base pretraining with the existing official training stack.

The model subtree in model.json is copied from the repository's verified v1.3
release checkpoint fixture. It includes all six supervision heads and their label
encodings. No checkpoint weights are loaded: a new run initializes randomly.

REPRODUCTION_ASSUMPTION: the report does not publish a complete run config.
Optimizer, masking, augmentation, token budget, seeds and checkpoint intervals
below follow scripts/official/v1_2/base.py. The location-query budget is an
explicit assumption. These defaults do not claim exact recovery of the final run.
See README.md in this directory for provenance, CLI overrides and local checks.
"""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

MODEL_CONFIG_PATH = Path(__file__).with_name("model.json")
MAP_MODALITIES = [
    "worldcover",
    "srtm",
    "openstreetmap_raster",
    "wri_canopy_height_map",
    "cdl",
    "worldcereal",
]

# REPRODUCTION_ASSUMPTION: values not specified in the report. The model's
# supervision weights/labels come from the released checkpoint, not assumptions.
REPRODUCTION_ASSUMPTIONS = {
    "optimizer": "v1.2 lr=1e-4, weight_decay=0.02; AdamW library defaults",
    "masking": "v1.2 random_time_with_decode ratios 0.5/0.5/0.5",
    "loss": "v1.2 token discrimination tau=0.1, same_target_threshold=0.999",
    "sampling": "v1.2 token_budget=2250, hw=1..12, patch_size=1..8, seed=3622",
    "location_queries": "uniform allowed patch divisors; max_location_queries=512",
    "training": "v1.2 microbatch=64, grad_clip=1, augmentation and init seed=12536",
    "checkpoint": "v1.2 permanent=5000, ephemeral=250 steps",
}


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the released v1.3 Base architecture, with random initialization."""
    model = Config.from_dict(json.loads(MODEL_CONFIG_PATH.read_text())["model"])
    assert isinstance(model, LatentMIMConfig)
    model.encoder_config.tokenization_config = common.tokenization_config
    model.decoder_config.tokenization_config = common.tokenization_config
    return model


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Reuse common launch and output handling with the v1.3 modalities."""
    common = build_common_components_default(script, cmd, run_name, cluster, overrides)
    model = Config.from_dict(json.loads(MODEL_CONFIG_PATH.read_text())["model"])
    common.training_modalities = list(model.encoder_config.supported_modality_names)
    common.tokenization_config = model.encoder_config.tokenization_config
    return common


def _masking_config(common: CommonComponents) -> MaskingConfig:
    """Use v1.2 masking as a REPRODUCTION_ASSUMPTION, with maps decode-only."""
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": MAP_MODALITIES,
        },
        tokenization_config=common.tokenization_config,
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Combine PixLatentMIM, six map heads and d128/d64 student losses."""
    cuda = torch.cuda.is_available()
    return LatentMIMTrainModuleConfig(
        # REPRODUCTION_ASSUMPTION: v1.2 optimizer and microbatch defaults.
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.02, fused=cuda),
        rank_microbatch_size=64,
        masking_config=_masking_config(common),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": MAP_MODALITIES,
            }
        ),
        token_exit_cfg={name: 0 for name in common.training_modalities},
        pixel_targets=True,
        # Frozen projection-only target: existing EMA mechanism must be a no-op.
        ema_decay=(1.0, 1.0),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        dp_config=(
            DataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
            )
            if cuda and dist.is_initialized()
            else None
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Use a single view and sample pooler patches that divide encoder patches."""
    return OlmoEarthDataLoaderConfig(
        work_dir=common.save_folder,
        global_batch_size=512,
        # REPRODUCTION_ASSUMPTION: v1.2 geometry/token budget and loader seed.
        token_budget=2250,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=1,
        max_patch_size=8,
        seed=3622,
        num_workers=16,
        prefetch_factor=4,
        num_masked_views=1,
        # REPRODUCTION_ASSUMPTION: not published in the v1.3 report.
        # Existing sampler enforces p_pool <= p_encoder and divisibility.
        max_location_queries=512,
        masking_config=_masking_config(common),
        tokenization_config=common.tokenization_config,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Read existing H5 samples; the data path is required as a CLI override."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/path/to/official/h5/sample_count",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Train for the reported number of steps and resume native checkpoints."""
    return (
        TrainerConfig(
            work_dir=common.save_folder,
            save_folder=common.save_folder,
            max_duration=Duration.steps(667200),
            load_strategy=LoadStrategy.if_available,
            checkpointer=CheckpointerConfig(work_dir=common.save_folder),
            metrics_collect_interval=10,
            cancel_check_interval=25,
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                project="olmoearth-v1_3",
                enabled=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "console_logger",
            ConsoleLoggerCallback(
                metrics=[
                    "train/*",
                    "supervision/*",
                    "student/*",
                    "optim/*",
                    "checkpoint/*",
                    "throughput/*",
                ]
            ),
        )
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        # REPRODUCTION_ASSUMPTION: v1.2 checkpoint intervals.
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=250,
            ),
        )
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
    )
