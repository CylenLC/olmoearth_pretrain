# Native OSS HDF5 access

Install the optional backend alongside the existing training dependencies:

```bash
uv sync --locked --extra training --extra oss --python 3.12
```

`OlmoEarthDatasetConfig` accepts an `s3://bucket/prefix/modalities/count` path
and an `OSSConfig`. The final directory name must still be the sample count;
keep the official directory layout, `sample_metadata.csv`,
`latlon_distribution.npy` and `sample_N.h5` filenames.

```python
from olmoearth_pretrain.data.dataset import OSSConfig, OlmoEarthDatasetConfig

dataset = OlmoEarthDatasetConfig(
    h5py_dir="s3://olmoearth/PREFIX/MODALITIES/3996",
    training_modalities=["sentinel2_l2a", "sentinel1", "latlon"],
    oss=OSSConfig(
        endpoint_url="https://s3.oss-cn-beijing.aliyuncs.com",
        region="cn-beijing",
        # profile="oss-training",  # optional AWS profile name, never credentials
    ),
).build()
dataset.prepare()
```

The shared training configuration accepts the equivalent CLI overrides:

```text
--dataset.h5py_dir=s3://olmoearth/PREFIX/MODALITIES/3996
--dataset.oss.endpoint_url=https://s3.oss-cn-beijing.aliyuncs.com
--dataset.oss.region=cn-beijing
```

Use the endpoint for the bucket's actual region. For subsequent ECS testing,
the documented internal endpoint is `https://s3.oss-REGION-internal.aliyuncs.com`.
The implementation uses virtual addressing and `signature_version="s3"`, matching
Alibaba's [Python S3 compatibility example](https://www.alibabacloud.com/help/en/oss/developer-reference/use-aws-sdks-to-access-oss).
It sets finite connection/read timeouts (10/60 seconds) and SDK retries (3),
in addition to s3fs's existing finite retry behavior. No extra retry loop is added.

## Credentials and processes

Use the standard AWS provider chain through s3fs: `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, optionally `AWS_SESSION_TOKEN`, or an AWS profile/provider.
If the current shell already has `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`,
map them in the launching process without printing or saving their values:

```bash
export AWS_ACCESS_KEY_ID="${OSS_ACCESS_KEY_ID:?OSS credential required}"
export AWS_SECRET_ACCESS_KEY="${OSS_ACCESS_KEY_SECRET:?OSS credential required}"
# For STS, also export AWS_SESSION_TOKEN from your temporary credential provider.
```

No credentials are accepted by `OSSConfig`. Do not put credentials in CLI arguments,
URIs, training configuration, checked-in `.env` files or logs. The endpoint must
be an HTTPS origin without userinfo, query parameters or fragments. Configs saved
by the trainer contain the endpoint, region, optional profile name and reader selection.

Alibaba ECS RAM Role credentials are **not** assumed to be part of the AWS default
credential chain. A future ECS deployment must provide a compatible refreshable
provider (for example, an AWS profile `credential_process` adapter for temporary
credentials) or inject temporary AWS environment variables. This Mac check does
not validate RAM Role discovery or STS refresh.

The official loader uses `spawn`. UPath preserves connection options when pickled
and creates a process-local filesystem/client lazily; it does not pickle a live
client. HDF5 and underlying file handles close at the end of each sample read.
Use the official worker lifecycle, not `fork` with an initialized S3 client.

## Behavior and resume

For Alibaba OSS HDF5 files, the default reader uses boto3 Full GET. Set
`--dataset.oss.use_fsspec=true` to use s3fs for the HDF5 objects. The DataLoader
uses `--data_loader.max_inflight_oss_gets_per_worker` for either backend. With the
value at 2, each worker has up to two downloads in flight and decodes buffered
objects in sampler order. The value 1 uses the synchronous reader. With s3fs
selected, that is a seekable `UPath.open("rb")` stream. Cached or throttled s3fs
reads use the normal per-sample reader.

Both modes use the same HDF5 parser, timestep masks, crops, normalization,
transforms and collator. CSV and NPY metadata use the filesystem options in
`OSSConfig`. Leave `dataset.cache_dir=None` when comparing native reads.

`filter_idx_file`, loader memmaps/work_dir and training output/checkpoint paths
retain their existing local behavior. Dataset fingerprints include the storage
path, so SSD and OSS have different fingerprints. Resume using the same URI and
filter file; moving a checkpoint between roots retains the official fingerprint
warning. No sampler/resume semantics are changed.

## Bounded local verification

The diagnostic requires 3–10 matching **existing local** H5 files. It never
downloads a local copy, lists the entire bucket, writes OSS objects or trains.
Selected files must total at most 100 MiB. Choose a fresh ignored output path:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --locked --extra training --extra oss python -m scripts.diagnostics.check_oss_dataset \
  --local-dir /LOCAL/PREFIX/MODALITIES/3996 \
  --oss-dir s3://olmoearth/PREFIX/MODALITIES/3996 \
  --endpoint-url https://s3.oss-cn-beijing.aliyuncs.com \
  --region cn-beijing --indices 0,233,1721,2316 \
  --output local_output/oss-native-check-01
```

It recursively checks HDF5 groups, datasets and attributes with strict dtype/shape
and value equality (including NaN/Inf positions); checks official Dataset output
with and without CutMix, normalization, NDVI and one-hot WorldCover; compares two
passes through the official shuffled/augmented/masked DataLoader at 0/1/2/4 workers;
and roundtrips loader state. Python, NumPy and Torch random states are reset for
each paired read. Different worker counts need not produce the same augmentation;
SSD and OSS must match exactly **within** each worker configuration.

`results.json` records the object set, successful range payload bytes (including
spawn workers), per-sample latency, first-batch and total loader timings, and CPU/
RSS observations. Instrumentation is confined to this diagnostic. Byte counts
exclude HTTP/TLS overhead and failed retries. Repeated calls open fresh file
handles but do not clear OS caches; this is not a cold-cache or ECS throughput
benchmark. Tiny sample sets are dominated by worker startup and iterator teardown;
total iteration time is not a measure of sustained loading throughput.

Offline regression tests:

```bash
uv run --locked --extra training --extra dev --extra oss --extra beaker pytest \
  tests/unit/data tests/integration/data tests/unit/dataset \
  tests/unit/test_config.py tests/unit/test_checkpoint_configs.py
```

## Nano short-run comparison

From the checkout and training environment, use the same saved single-GPU Nano
fsspec config for both arms. Provide credentials through the AWS provider chain.
The runner requires an idle GPU, caps the run at 24 steps, disables W&B,
evaluation and checkpoints, and requires a fresh output directory per arm.

```bash
export PYTHONPATH="$PWD"
python benchmarks/oss_h5/nano_ab.py "$nano_config" local_output/nano-oss-a --prefetch off
python benchmarks/oss_h5/nano_ab.py "$nano_config" local_output/nano-oss-b --prefetch on
python benchmarks/oss_h5/summarize.py local_output/nano-oss-a local_output/nano-oss-b
```

The summary includes the first-batch wait and steps 5–24 as the warm window.
HDF5 dataset-read time subtracts s3fs range-fetch time and includes local copies
as well as decompression. Worker request and payload counts include work fetched
ahead of consumption. Summed process RSS can count shared pages more than once.

Run `benchmarks/oss_h5/h5_crop.py --help` for the paired full-read and hyperslab
comparison on scratch copies of up to ten HDF5 files. The optional OSS arguments
measure s3fs range-cache activity for the corresponding remote objects.
