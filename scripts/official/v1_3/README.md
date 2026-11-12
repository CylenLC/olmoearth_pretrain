# OlmoEarth v1.3 pretraining

`base.py` configures the existing Dataset, LatentMIMTrainModule and OLMo-core
Trainer. New runs initialize randomly. The training objective includes
PixLatentMIM Lite, six map supervision losses and both cosine/Gram distillation
losses for the d128 student and its d64 prefix. There is one masked view and no
instance contrastive loss. `projection_only_target=True`, `pixel_targets=True`
and `ema_decay=(1, 1)` use the existing frozen projection-only target mechanism.

## Provenance and reproduction assumptions

The v1.3 [report](https://allenai.org/papers/olmoearth-v1-3) specifies ViT-Base,
d768 location tokens, d128/d64 students, a 256-wide back-projection MLP, 2048
Gram tokens, AdamW, batch 512, 667200 steps, 8000 linear warmup steps and cosine
annealing.

`model.json` is copied verbatim from
`tests/fixtures/checkpoint_configs/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1.json`.
The fixture's `_source` records verification against the released v1.3 Base
checkpoint. It supplies architecture, tokenization and supervision settings that
are absent from the paper: four Attention Pooler blocks, encoder mixed 3D RoPE
with base 10000, decoder 2D RoPE, and the following heads. All labels are already
present in the official normalized H5 data; no external teacher or precomputed
features are required.

| Map | Task | Loss and weight |
| --- | --- | --- |
| CDL | 133 classes, checkpoint label mapping | Cross entropy, 0.1 |
| WorldCover | 11 classes, checkpoint label mapping | Cross entropy, 0.1 |
| OSM | 30 binary channels | Frequency-weighted BCE, 0.1 |
| WorldCereal | 8 binary channels | Frequency-weighted BCE, 0.1 |
| SRTM | 1 continuous channel | L1, 1.0 |
| Canopy Height | 1 continuous channel | L1, 1.0 |

The full training-run configuration is not published in that fixture.
`REPRODUCTION_ASSUMPTION` marks the following defaults in `base.py`; this recipe
does not claim exact reconstruction of the authors' run:

- From the official v1.2 recipe: LR 1e-4, weight decay 0.02, default AdamW
  betas/epsilon, microbatch 64, gradient clipping 1.0, masking ratios 0.5/0.5/0.5,
  token discrimination temperature 0.1 and same-target threshold 0.999,
  token budget 2250, spatial token side 1..12, encoder patch size 1..8, loader
  seed 3622 and initialization seed 12536. Unspecified data and augmentation
  options retain the official builders' defaults.
- Pooler patch sampling uses the existing loader: uniformly draw an allowed
  divisor of encoder patch size, with a 512-query budget. This enforces
  `p_pool <= p_encoder` and `p_encoder % p_pool == 0`. The original budget and
  distribution are not disclosed. As documented by the loader, the equal-patch
  case is always allowed, even if its query count exceeds the budget.
- Library defaults set cosine minimum LR and unspecified optimizer options;
  these remain assumptions, rather than reported v1.3 hyperparameters.
- Permanent checkpoint interval 5000 and ephemeral interval 250 follow v1.2.
  Precision/distribution follow v1.2 on distributed CUDA; single-device execution
  uses the available default device without FSDP or fused CUDA AdamW on a Mac.

## Environment

Run from the repository root. Python 3.11–3.13 and PyTorch 2.9 are required by
the project. This creates an independent environment for this checkout:

```bash
uv venv --python 3.12 .venv
uv sync --locked --extra training --extra beaker --extra dev
```

The generic official entrypoint currently imports the Beaker launch helpers even
for local execution, hence the `beaker` extra. Neither login nor cloud logging is
required. W&B is disabled by default and downstream evaluation datasets are not
required for this recipe.

## Start and resume

Set `H5_DIR` to the existing numeric sample directory containing `sample_*.h5`,
`sample_metadata.csv` and `latlon_distribution.npy`, not its parent directory.
This command does not download, copy or rewrite the input dataset.

```bash
uv run --locked --extra training --extra beaker torchrun --standalone \
  --nproc_per_node=8 scripts/official/v1_3/base.py train v1_3_base local \
  --dataset.h5py_dir="$H5_DIR" \
  --common.save_folder=./local_output/v1_3_base
```

For single-device CPU/MPS/CUDA execution, replace `torchrun ...` with `python`
and `train` with `train_single`. Full Base pretraining is intended for CUDA
hardware. To override batch size or duration, add:

```bash
--data_loader.global_batch_size=2 --train_module.rank_microbatch_size=2 \
--trainer.max_duration.value=3
```

Batch size must be divisible by microbatch size times data-parallel world size.
Use `--common.save_folder=...` to update all output/work/checkpointer paths
together. `dry_run` replaces `train` to inspect the resolved configuration.

Native Trainer checkpoints resume automatically from the same output directory;
increase `--trainer.max_duration.value` to continue. Use a fresh output directory
or `--trainer.load_strategy=never` for a new random initialization. A converted
`weights.pth` artifact contains no optimizer or loader progress and is not a
native training resume checkpoint. Partial initialization from v1.2 weights is
not implemented by this entrypoint.

## Bounded local validation

The validation utility detects the sample directory below `--h5-root`, selects
a few real samples covering all nine modalities without modifying their files,
instantiates the full Base model, then uses a smaller encoder/decoder with the
same d768 location tokens, d128/d64 students and all six supervision heads.
All generated files and native checkpoints go under ignored `local_output/`.

```bash
uv run --locked --extra training --extra beaker --extra dev python \
  -m scripts.official.v1_3.validate \
  --h5-root /path/to/h5py_data_w_missing_timesteps_zstd_3_128_x_4 \
  --output-dir ./local_output/v1_3_validation
```

The utility checks the real batch, finite component losses, gradients, optimizer
updates and a native checkpoint restore followed by a further training step.
It prints hardware capabilities and supports `--device cpu|mps|cuda|auto`.
An unsupported MPS operation must be reported and rerun explicitly on CPU;
the objectives and model components are retained on both devices.
