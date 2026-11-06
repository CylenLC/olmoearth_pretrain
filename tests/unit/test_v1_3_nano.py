"""Tests for the custom capacity-scaled v1.3 Nano recipe."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.official.v1_3 import base, nano  # noqa: E402


def _common() -> SimpleNamespace:
    model = base.build_model_config(SimpleNamespace(tokenization_config=None))
    return SimpleNamespace(tokenization_config=model.encoder_config.tokenization_config)


def test_nano_overrides_capacity_without_changing_v13_method() -> None:
    """Nano changes capacity fields while retaining v1.3-specific methods."""
    common = _common()
    base_config = base.build_model_config(common)
    nano_config = nano.build_model_config(common)

    assert (
        base_config.encoder_config.embedding_size,
        base_config.encoder_config.depth,
        base_config.encoder_config.num_heads,
    ) == (768, 12, 12)
    assert (
        nano_config.encoder_config.embedding_size,
        nano_config.encoder_config.depth,
        nano_config.encoder_config.num_heads,
        nano_config.encoder_config.mlp_ratio,
        nano_config.encoder_config.patch_embed_hidden_sizes,
    ) == (128, 4, 8, 4.0, [12])
    assert (
        nano_config.decoder_config.encoder_embedding_size,
        nano_config.decoder_config.decoder_embedding_size,
        nano_config.decoder_config.depth,
        nano_config.decoder_config.num_heads,
        nano_config.decoder_config.mlp_ratio,
    ) == (128, 128, 4, 8, 4.0)

    pooler = nano_config.encoder_config.attention_pooler_config
    assert pooler is not None
    assert pooler.pooler_dim == 128
    assert pooler.attn_dim == 128
    assert pooler.resolved_num_heads(8) == 8
    assert pooler.pooler_depth == 4
    assert pooler.student_dims == [128, 64]
    assert nano_config.encoder_config.position_encoding == "rope_3d_mixed"
    assert nano_config.decoder_config.position_encoding == "rope"
    assert nano_config.projection_only_target is True
    assert nano_config.location_token_distillation_head_config is not None
    assert nano_config.supervision_head_config is not None
    assert set(nano_config.supervision_head_config.modality_configs) == set(
        base.MAP_MODALITIES
    )


def test_nano_builds_v13_modules_with_valid_attention_shapes() -> None:
    """The v1.3 modules instantiate with Nano attention head dimensions."""
    model = nano.build_model_config(_common()).build()

    assert len(model.encoder.blocks) == 4
    assert len(model.decoder.blocks) == 4
    assert model.encoder.blocks[0].attn.head_dim == 16
    assert model.decoder.blocks[0].attn.head_dim == 16

    pooler = model.encoder.attention_pooler
    assert pooler is not None
    assert len(pooler.read_blocks) == 4
    assert len(pooler.self_attn_blocks) == 4
    assert pooler.read_blocks[0].attn.head_dim == 16
    assert pooler.student_dims == [128, 64]
    assert model.location_token_distillation_head is not None
    assert model.supervision_head is not None
    assert set(model.supervision_head.heads) == set(base.MAP_MODALITIES)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 0 < parameter_count < 238_753_536


def test_nano_reuses_v13_training_method_and_optimizer() -> None:
    """Nano uses v1.3 losses without the v1.2 contrastive loss."""
    common = _common()
    common.training_modalities = list(
        nano.build_model_config(common).encoder_config.supported_modality_names
    )
    train_config = base.build_train_module_config(common)

    assert type(train_config).__name__ == "LatentMIMTrainModuleConfig"
    assert train_config.pixel_targets is True
    assert train_config.ema_decay == (1.0, 1.0)
    assert train_config.optim_config.lr == 1e-4
    assert train_config.optim_config.weight_decay == 0.02
    assert train_config.rank_microbatch_size == 64
    assert train_config.loss_config.loss_config["type"] == (
        "modality_patch_discrimination_masked_negatives_vec"
    )
    assert not hasattr(train_config, "contrastive_config")
