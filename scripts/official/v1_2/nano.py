"""v1.2 nano: ViT-nano encoder + mixed 3D RoPE, decoder depth 4."""

import logging
import os
import sys
from functools import partial
from pathlib import Path

from base import build_common_components as build_common_components_base
from base import build_dataloader_config as build_dataloader_config_base
from base import build_dataset_config as build_dataset_config_base
from base import (
    build_size_model_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_train_module_config as build_train_module_config_base,
)
from olmo_core.optim import AdamWConfig

from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoaderConfig,
    _set_cpu_affinity,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig, OSSConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.02
# Nano's hidden size is small, so use a small patch-embed hidden width.
PATCH_EMBED_HIDDEN_SIZES = [12]


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    config = build_train_module_config_base(common)
    config.optim_config = AdamWConfig(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=False,
    )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    return build_size_model_config(common, "nano", PATCH_EMBED_HIDDEN_SIZES)


_ALIYUN_FULL_GET_PROFILES = ("aliyun", "aliyun-cpu-isolated")
_TRAINING_CPUS = tuple(range(4))
_WORKER_CPUS = tuple(range(4, 16))

_ALIYUN_DATASET = (
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Run the Alibaba storage profiles using the local launcher."""
    if cluster in (*_ALIYUN_FULL_GET_PROFILES, "aliyun-ossfs2"):
        cluster = "local"
    return build_common_components_base(script, cmd, run_name, cluster, overrides)


def build_dataset_config(
    common: CommonComponents, *, cluster: str = ""
) -> OlmoEarthDatasetConfig:
    """Use Full GET on Alibaba ECS and expose the mounted dataset as a fallback."""
    config = build_dataset_config_base(common)
    if cluster in _ALIYUN_FULL_GET_PROFILES:
        config.h5py_dir = f"s3://olmoearth/{_ALIYUN_DATASET}"
        config.oss = OSSConfig(
            endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com",
            region="cn-beijing",
            use_fsspec=True,
            fsspec_full_get=True,
        )
    elif cluster == "aliyun-ossfs2":
        config.h5py_dir = f"/mnt/olmoearth/{_ALIYUN_DATASET}"
    return config


def build_dataloader_config(
    common: CommonComponents, *, cluster: str = ""
) -> OlmoEarthDataLoaderConfig:
    """Use the measured prefetch settings for the default Alibaba profile."""
    config = build_dataloader_config_base(common)
    if cluster in _ALIYUN_FULL_GET_PROFILES:
        config.prefetch_factor = 2
        config.max_inflight_oss_gets_per_worker = 2
    if cluster == "aliyun-cpu-isolated":
        config.worker_cpu_affinity = list(_WORKER_CPUS)
    return config


def _isolate_training_cpus() -> None:
    """Verify the measured single-rank ECS layout before narrowing affinity."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("aliyun-cpu-isolated supports a single training rank")
    layout = (*_TRAINING_CPUS, *_WORKER_CPUS)
    if sys.platform != "linux" or not set(layout) <= os.sched_getaffinity(0):
        raise ValueError("aliyun-cpu-isolated requires Linux with CPUs 0–15 available")
    for cpu in layout:
        topology = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        first = cpu // 2 * 2
        if topology.read_text().strip() not in (
            f"{first}-{first + 1}",
            f"{first},{first + 1}",
        ):
            raise ValueError("aliyun-cpu-isolated requires adjacent SMT sibling pairs")
    _set_cpu_affinity(_TRAINING_CPUS)


def run() -> None:
    """Run the experiment."""
    cluster = sys.argv[3] if len(sys.argv) > 3 else ""
    if cluster == "aliyun-cpu-isolated" and sys.argv[1] in ("train", "train_single"):
        _isolate_training_cpus()
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=partial(build_dataset_config, cluster=cluster),
        dataloader_config_builder=partial(build_dataloader_config, cluster=cluster),
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
