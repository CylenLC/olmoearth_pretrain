"""Custom v1.3 Tiny recipe with the v1.2 Tiny encoder capacity."""

from __future__ import annotations

from olmoearth_pretrain.data.dataset import OSSConfig
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from scripts.official.v1_3 import base, nano


def build_model_config(common):
    """Retain v1.3 objectives and scale its model to Tiny."""
    model = nano.build_model_config(common)
    size = MODEL_SIZE_ARGS["tiny_shallow_decoder"]
    encoder = model.encoder_config
    decoder = model.decoder_config
    pooler = encoder.attention_pooler_config
    encoder.embedding_size = size["encoder_embedding_size"]
    encoder.depth = size["encoder_depth"]
    encoder.num_heads = size["encoder_num_heads"]
    encoder.patch_embed_hidden_sizes = [64]
    pooler.pooler_dim = encoder.embedding_size
    pooler.attn_dim = encoder.embedding_size
    decoder.encoder_embedding_size = encoder.embedding_size
    decoder.decoder_embedding_size = size["decoder_embedding_size"]
    decoder.depth = size["decoder_depth"]
    decoder.num_heads = size["decoder_num_heads"]
    decoder.pooler_dim = pooler.pooler_dim
    model.validate()
    encoder.validate()
    decoder.validate()
    pooler.validate(
        encoder_num_heads=encoder.num_heads, position_encoding=encoder.position_encoding
    )
    return model


def build_dataset_config(common):
    """Use the established Alibaba Full GET dataset."""
    config = base.build_dataset_config(common)
    config.h5py_dir = (
        "s3://olmoearth/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
        "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
        "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
    )
    config.oss = OSSConfig(
        endpoint_url="https://oss-cn-beijing-internal.aliyuncs.com",
        region="cn-beijing",
        use_fsspec=True,
        fsspec_full_get=True,
    )
    return config


def build_dataloader_config(common):
    """Configure bounded Full GET prefetch."""
    config = base.build_dataloader_config(common)
    config.prefetch_factor = 2
    config.max_inflight_oss_gets_per_worker = 2
    return config


def run():
    """Launch the v1.3 Tiny recipe."""
    base.main(
        common_components_builder=base.build_common_components,
        model_config_builder=build_model_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        train_module_config_builder=base.build_train_module_config,
        trainer_config_builder=base.build_trainer_config,
    )


if __name__ == "__main__":
    run()
