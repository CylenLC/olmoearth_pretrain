"""Bounded real-H5 integration checks for the v1.3 recipe (no data writes)."""

import argparse
import copy
import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from olmo_core.train.callbacks import Callback
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.utils import seed_all

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.internal.experiment import CommonComponents, build_config
from scripts.official.v1_3 import base


def find_h5_dir(root: Path) -> Path:
    """Require one unambiguous existing sample directory beneath the given root."""
    candidates = (
        [root / "sample_metadata.csv"]
        if (root / "sample_metadata.csv").is_file()
        else sorted(root.rglob("sample_metadata.csv"))
    )
    candidates = [p.parent for p in candidates if (p.parent / "sample_0.h5").is_file()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one H5 sample directory, found {candidates}"
        )
    return candidates[0].resolve()


def _gradient_checks(model: torch.nn.Module) -> dict[str, int]:
    """Require finite, nonzero gradients in every enabled trainable component."""
    groups = {
        "vit": [
            p
            for n, p in model.named_parameters()
            if n.startswith("encoder.") and "attention_pooler" not in n
        ],
        "pooler": [
            p
            for n, p in model.named_parameters()
            if n.startswith("encoder.attention_pooler.") and ".student." not in n
        ],
        "student": list(model.encoder.attention_pooler.student.parameters()),
        "decoder": list(model.decoder.parameters()),
    }
    for name, head in model.supervision_head.heads.items():
        groups[f"supervision/{name}"] = list(head.parameters())
    for dim, head in model.location_token_distillation_head.back_projections.items():
        groups[f"distillation/d{dim}"] = list(head.parameters())
    result = {}
    for name, params in groups.items():
        gradients = [p.grad for p in params if p.grad is not None]
        assert gradients, f"No gradients in {name}"
        assert all(torch.isfinite(g).all().item() for g in gradients), name
        count = sum(torch.count_nonzero(g).item() for g in gradients)
        assert count > 0, f"Only zero gradients in {name}"
        result[name] = int(count)
    return result


@dataclass
class ValidationCallback(Callback):
    """Fail on nonfinite training metrics or disconnected model components."""

    metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    gradient_steps: list[int] = field(default_factory=list)
    step_losses: dict[int, float] = field(default_factory=dict)

    def pre_optim_step(self) -> None:
        """Check the actual Trainer backward pass before optimizer execution."""
        module = self.trainer.train_module
        loss = self.trainer.get_metric(f"train/{module.total_loss_name}")
        assert loss is not None and torch.isfinite(loss).all().item()
        self.step_losses[self.step] = float(loss)
        _gradient_checks(self.trainer.train_module.model)
        self.gradient_steps.append(self.step)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Keep local evidence of the finite losses from the training loop."""
        for key, value in metrics.items():
            if key.startswith("train/"):
                assert math.isfinite(value), (step, key, value)
        self.metrics.setdefault(step, {}).update(metrics)


def validate(
    h5_root: Path, output_dir: Path, device_name: str = "auto"
) -> dict[str, Any]:
    """Exercise the real model/loss/Trainer/checkpointer stack on two H5 samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "train").exists():
        raise ValueError(
            "Use a fresh output directory so an old checkpoint cannot pass this check"
        )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    capabilities = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if device_name == "auto":
        device_name = (
            "cuda" if capabilities["cuda"] else "mps" if capabilities["mps"] else "cpu"
        )
    device = torch.device(device_name)
    print(f"Capabilities: {capabilities}; validation device: {device}", flush=True)
    seed_all(12536)
    h5_dir = find_h5_dir(h5_root)
    template = Config.from_dict(json.loads(base.MODEL_CONFIG_PATH.read_text())["model"])
    modalities = template.encoder_config.supported_modality_names
    metadata = pd.read_csv(h5_dir / "sample_metadata.csv")
    complete = metadata.loc[(metadata[modalities].fillna(0) > 0).all(axis=1)]
    if len(complete) < 2:
        raise ValueError(
            "Need at least two real samples containing all nine training modalities"
        )
    selected = complete["sample_index"].iloc[:2].to_numpy(dtype=np.int64)
    inputs = [h5_dir / f"sample_{i}.h5" for i in selected]
    inputs += [h5_dir / "sample_metadata.csv", h5_dir / "latlon_distribution.npy"]
    mtimes = {str(p): p.stat().st_mtime_ns for p in inputs}
    filter_path = output_dir / "sample_indices.npy"
    np.save(filter_path, selected)
    print(f"H5 directory: {h5_dir}; selected samples: {selected.tolist()}", flush=True)

    common = CommonComponents(
        "v1_3_validation",
        str(output_dir / "train"),
        list(modalities),
        tokenization_config=template.encoder_config.tokenization_config,
    )
    config = build_config(
        common=common,
        model_config_builder=base.build_model_config,
        dataset_config_builder=base.build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        train_module_config_builder=base.build_train_module_config,
        trainer_config_builder=base.build_trainer_config,
        overrides=[],
    )
    config.model.validate()
    full_model = config.model.build()
    full_parameters = sum(p.numel() for p in full_model.parameters())
    print(f"Full Base instantiated: {full_parameters:,} parameters", flush=True)
    del full_model
    gc.collect()

    # Validation-only scale reduction. Keep d768 -> d128/d64, all maps/losses,
    # the pixel target and a nontrivial read/self-attention pooler intact.
    encoder = config.model.encoder_config
    encoder.embedding_size = 48
    encoder.num_heads = 4
    encoder.depth = 1
    encoder.min_patch_size = encoder.max_patch_size = 2
    encoder.attention_pooler_config.attn_dim = 48
    encoder.attention_pooler_config.pooler_depth = 2
    decoder = config.model.decoder_config
    decoder.encoder_embedding_size = decoder.decoder_embedding_size = 48
    decoder.num_heads = 4
    decoder.depth = 1
    config.dataset.h5py_dir = str(h5_dir)
    config.dataset.filter_idx_file = str(filter_path)
    dl = config.data_loader
    dl.global_batch_size = 2
    dl.num_workers = 0
    dl.prefetch_factor = None
    dl.min_patch_size = dl.max_patch_size = 2
    dl.sampled_hw_p_list = [4]
    dl.max_timesteps = 2
    dl.token_budget = 256
    dl.shuffle = False
    dl.target_device_type = device.type
    config.train_module.rank_microbatch_size = 2
    config.train_module.dp_config = None
    config.train_module.optim_config.fused = False
    config.trainer.device = device.type
    config.trainer.max_duration = Duration.steps(2)
    config.trainer.metrics_collect_interval = 1
    config.trainer.load_strategy = LoadStrategy.never
    cp = config.trainer.callbacks["checkpointer"]
    cp.save_interval = 1
    cp.ephemeral_save_interval = None
    # Match the formal Trainer's initial checkpoint, including its optimizer
    # state initialization; the forward/optimizer unit check is a separate stack.
    cp.pre_train_checkpoint = None
    cp.save_async = False
    (output_dir / "config.json").write_text(
        json.dumps(config.as_config_dict(), indent=2)
    )

    def build_stack():
        model = config.model.build().to(device)
        module = config.train_module.build(model, device=device)
        dataset = config.dataset.build()
        loader = config.data_loader.build(dataset)
        return model, module, loader

    model, module, loader = build_stack()
    loader.reshuffle(1, in_memory=True)
    batch = next(iter(loader))
    patch_size, sample, pool_patch = batch
    assert pool_patch is not None and 1 <= pool_patch <= patch_size
    assert patch_size % pool_patch == 0
    assert sample.batch_size == 2
    print(
        f"Real batch loaded: patch={patch_size}, pooler patch={pool_patch}", flush=True
    )
    # Exercise the finer output grid even if this draw selected equal patches.
    loss, latent, decoded, target, extra = module.model_forward(
        sample.to_device(device),
        patch_size,
        module.token_exit_cfg,
        location_query_patch_size=1,
    )
    required = {f"supervision/{m}" for m in base.MAP_MODALITIES}
    required |= {
        f"student/distill_{term}_d{dim}"
        for term in ("cosine", "gram")
        for dim in (64, 128)
    }
    assert extra is not None and required <= extra.keys(), (
        required - (extra or {}).keys()
    )
    component_losses = {key: float(extra[key]) for key in sorted(required)}
    component_losses["mim"] = float(module.loss_fn(decoded, target).detach())
    assert torch.isfinite(loss).all().item()
    assert all(math.isfinite(value) for value in component_losses.values())
    loss.backward()
    gradients = _gradient_checks(model)
    before = model.encoder.attention_pooler.student[0].weight.detach().clone()
    module.optimizer.step()
    assert not torch.equal(before, model.encoder.attention_pooler.student[0].weight)
    module.optimizer.zero_grad(set_to_none=True)
    print(
        f"Forward/backward/optimizer passed: total={float(loss.detach()):.6f}; {component_losses}",
        flush=True,
    )
    del loss, latent, decoded, target, batch, sample, loader, model, module
    gc.collect()

    # Train from random initialization, independently of the manual check above.
    model, module, loader = build_stack()
    callback = ValidationCallback()
    config.trainer.callbacks["validation"] = callback
    trainer = config.trainer.build(module, loader)
    trainer.callbacks["config_saver"].config = config.as_config_dict()
    trainer.fit()
    assert trainer.global_step == 2 and callback.gradient_steps == [1, 2]
    checkpoint_path = str(trainer.checkpointer.latest_checkpoint(trainer.save_folder))
    saved_model = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    saved_optimizer = copy.deepcopy(module.optimizer.state_dict())
    saved_loader = copy.deepcopy(loader.state_dict())

    # New model/optimizer/loader: verify native restore, rather than loading only weights.
    resumed_model, resumed_module, resumed_loader = build_stack()
    resume_callback = ValidationCallback()
    config.trainer.callbacks["validation"] = resume_callback
    config.trainer.load_strategy = LoadStrategy.if_available
    config.trainer.max_duration = Duration.steps(3)
    resumed = config.trainer.build(resumed_module, resumed_loader)
    resumed.callbacks["config_saver"].config = config.as_config_dict()
    resumed.load_checkpoint(
        checkpoint_path, load_trainer_state=True, load_optim_state=True
    )
    assert resumed.global_step == 2
    assert resumed_loader.state_dict() == saved_loader
    for key, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved_model[key], rtol=0, atol=0)
    actual_optimizer = resumed_module.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == saved_optimizer["param_groups"]
    for key, state in saved_optimizer["state"].items():
        for field_name, value in state.items():
            actual = actual_optimizer["state"][key][field_name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual.cpu(), value.cpu(), rtol=0, atol=0)
            else:
                assert actual == value
    resumed.fit()
    assert resumed.global_step == 3 and resume_callback.gradient_steps == [3]
    assert all(Path(path).stat().st_mtime_ns == stamp for path, stamp in mtimes.items())
    result = {
        "status": "passed",
        "device": device.type,
        "capabilities": capabilities,
        "h5_dir": str(h5_dir),
        "selected_samples": selected.tolist(),
        "full_base_parameters": full_parameters,
        "component_losses": component_losses,
        "nonzero_gradients": gradients,
        "checkpoint": checkpoint_path,
        "checkpoint_restore": "exact model, optimizer and loader state; step 2 -> 3",
        "training_metrics": callback.metrics,
        "resumed_metrics": resume_callback.metrics,
        "training_step_losses": callback.step_losses,
        "resumed_step_losses": resume_callback.step_losses,
        "input_mtimes_unchanged": True,
    }
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("local_output/v1_3_validation")
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    args = parser.parse_args()
    validate(args.h5_root, args.output_dir.resolve(), args.device)
