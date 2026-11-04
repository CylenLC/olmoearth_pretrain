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
by the trainer contain only endpoint, region and optional profile name.

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

Both backends use the same `UPath.open("rb")` → `h5py.File(file, "r")` path,
preprocessing, normalization, derived modalities, transforms and collator. CSV
and NPY metadata use file objects with the same filesystem options. There are no
new read modes, download paths, full-file buffers, caches or hot-path checks.
The existing optional Dataset cache stays disabled by default; leave
`dataset.cache_dir=None` when testing native reads.

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
