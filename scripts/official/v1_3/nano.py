"""Custom v1.3 Nano recipe built from the released v1.3 Base configuration.

This is a capacity-scaled research variant, not an officially released v1.3
checkpoint or model size. It retains the v1.3 LatentMIM model, RoPE variants,
Attention Pooler, location-token students/distillation, pixel targets and map
supervision, and applies the v1.2 Nano encoder/decoder size tuple.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from scripts.official.v1_3 import base  # noqa: E402

if TYPE_CHECKING:
    from olmoearth_pretrain.internal.experiment import CommonComponents


NANO_PATCH_EMBED_HIDDEN_SIZES = [12]


def apply_nano_capacity(model: LatentMIMConfig) -> LatentMIMConfig:
    """Apply only Nano capacity fields to an existing v1.3 model config.

    The v1.3 Attention Pooler has independent attention and residual widths. Both
    are reduced with the encoder width so its 8-head, 2D-RoPE attention remains
    shape-compatible. Its depth and student dimensions remain the v1.3 values.
    """
    size = MODEL_SIZE_ARGS["nano"]
    encoder = model.encoder_config
    decoder = model.decoder_config
    pooler = encoder.attention_pooler_config
    if pooler is None:
        raise ValueError("v1.3 Nano requires the v1.3 Attention Pooler")

    encoder.embedding_size = size["encoder_embedding_size"]
    encoder.depth = size["encoder_depth"]
    encoder.num_heads = size["encoder_num_heads"]
    encoder.mlp_ratio = size["mlp_ratio"]
    encoder.patch_embed_hidden_sizes = list(NANO_PATCH_EMBED_HIDDEN_SIZES)

    pooler.pooler_dim = size["encoder_embedding_size"]
    pooler.attn_dim = size["encoder_embedding_size"]

    decoder.encoder_embedding_size = size["encoder_embedding_size"]
    decoder.decoder_embedding_size = size["decoder_embedding_size"]
    decoder.depth = size["decoder_depth"]
    decoder.num_heads = size["decoder_num_heads"]
    decoder.mlp_ratio = size["mlp_ratio"]
    decoder.pooler_dim = pooler.pooler_dim

    # Run the same structural validators used by the official model builders.
    model.validate()
    encoder.validate()
    decoder.validate()
    pooler.validate(
        encoder_num_heads=encoder.num_heads,
        position_encoding=encoder.position_encoding,
    )

    encoder_head_dim = encoder.embedding_size // encoder.num_heads
    pooler_head_dim = (
        pooler.attn_dim or pooler.pooler_dim
    ) // pooler.resolved_num_heads(encoder.num_heads)
    decoder_head_dim = decoder.decoder_embedding_size // decoder.num_heads
    if (encoder_head_dim, pooler_head_dim, decoder_head_dim) != (16, 16, 16):
        raise ValueError(
            "v1.2 Nano head dimensions are incompatible with the retained v1.3 "
            f"RoPE/pooler modules: encoder={encoder_head_dim}, "
            f"pooler={pooler_head_dim}, decoder={decoder_head_dim}"
        )
    if pooler.student_dims != [128, 64]:
        raise ValueError(
            "The v1.3 location-token students must retain dimensions [128, 64], "
            f"got {pooler.student_dims}"
        )
    if not model.projection_only_target:
        raise ValueError("v1.3 Nano must retain projection_only_target=True")

    return model


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build v1.3 Base's released model config, then scale its capacity to Nano."""
    return apply_nano_capacity(base.build_model_config(common))


def run() -> None:
    """Run the shared v1.3 training stack with the Nano model config."""
    base.main(
        common_components_builder=base.build_common_components,
        model_config_builder=build_model_config,
        dataset_config_builder=base.build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        train_module_config_builder=base.build_train_module_config,
        trainer_config_builder=base.build_trainer_config,
    )


if __name__ == "__main__":
    run()
