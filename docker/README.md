# ABot-World Docker

Run ABot-World in a reproducible container instead of installing the
environment manually. The image follows the [Setup](../README.md#setup)
section of the main README:

- Ubuntu 22.04, CUDA 12.8, Python 3.12 (conda)
- PyTorch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0 (cu128)
- FlashAttention 2.8.1 (prebuilt wheel)
- SageAttention (built from source)
- `lightx2v_kernel` (built from source against CUTLASS)
- All Python dependencies from `requirements.txt`

> The image contains **only the runtime environment** — the ABot-World
> repository is mounted into the container at runtime, so always launch it
> from a local clone of this repo (e.g. via `docker/run.sh`). Code changes on
> the host take effect immediately without rebuilding the image.

## Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An NVIDIA GPU with a driver supporting CUDA 12.8 (tested on RTX 5090; see [FAQ.md](../FAQ.md) for RTX 3090/4090 notes)
- At least 64 GB of system RAM recommended

## 1. Get the image

### Use the prebuilt image (recommended)

A prebuilt environment image is available on Docker Hub — no compilation needed:

```bash
docker pull amapcvlab/abot-world:v0-env
```

Then pass it to `run.sh` via `IMAGE` (see step 3): `IMAGE=amapcvlab/abot-world:v0-env bash docker/run.sh`

### Or build it yourself

From the repository root:

```bash
bash docker/build.sh
```

Or directly:

```bash
docker build -f docker/Dockerfile -t abot-world:latest .
```

Build options (passed as environment variables to `build.sh` or as
`--build-arg` to `docker build`):

| Option | Default | Description |
| --- | --- | --- |
| `TORCH_CUDA_ARCH_LIST` | `8.6;8.9;9.0;12.0` | CUDA architectures compiled for SageAttention. Use `12.0` to build for RTX 5090 only (faster build). |
| `MAX_JOBS` | `4` | Parallel compilation jobs for the source builds. SageAttention's build parallelism multiplies internally and each compile unit can peak at 3-6 GB RAM; raise this only on hosts with plenty of RAM (OOM will hang the machine). |

> Note: `lightx2v_kernel` is compiled for `sm_120a` (Blackwell, e.g. RTX 5090)
> only, following the upstream LightX2V build configuration.

## 2. Download checkpoints

Checkpoints are not baked into the image. Download them to `./checkpoints`
on the host (they are mounted into the container by `docker/run.sh`):

```bash
# HuggingFace
hf download acvlab/ABot-World-0-5-5B-Iris --local-dir ./checkpoints/ABot-World-0-5-5B-Iris

# or ModelScope
modelscope download "amap_cvlab/ABot-World-0-5-5B-Iris" --local_dir ./checkpoints/ABot-World-0-5-5B-Iris
```

If you prefer to download inside the container (both `huggingface_hub` and
`modelscope` are preinstalled):

```bash
bash docker/run.sh bash
hf download acvlab/ABot-World-0-5-5B-Iris --local-dir ./checkpoints/ABot-World-0-5-5B-Iris
```

## 3. Run the Gradio demo

```bash
bash docker/run.sh
```

To use FantasyDecoder, enter the container with `bash docker/run.sh bash`,
then run `ABOTWORLD_CONFIG=configs/iris/model_inference.yaml bash web_client/run.sh`.
See the [model table](../README.md#models) for optional Iris-BiT weights.

Then open `http://localhost:2233` in your browser (Chrome recommended).

Runtime options:

```bash
CUDA_ID=1 bash docker/run.sh        # select a GPU
PORT=8080 bash docker/run.sh        # map a different host port
IMAGE=abot-world:v0.1 bash docker/run.sh
bash docker/run.sh bash             # interactive shell inside the container
```

Equivalent raw `docker run` command:

```bash
docker run --rm -it --gpus all \
  --ipc=host --shm-size=32GB \
  --ulimit nofile=65536:65536 \
  -v "$(pwd):/workspace/ABot-World" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 2233:2233 \
  abot-world:latest
```

## Mounted volumes

| Host path | Container path | Purpose |
| --- | --- | --- |
| `.` (repo root) | `/workspace/ABot-World` | Project code, `checkpoints/`, `outputs/` |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace cache (tokenizers, downloads) |

## Troubleshooting

- **`could not select device driver "nvidia"`** — the NVIDIA Container
  Toolkit is not installed or Docker was not restarted after installation.
- **CUDA errors from `lightx2v_kernel` on non-Blackwell GPUs** — the
  quantized kernels target `sm_120a`; see [FAQ.md](../FAQ.md) for hardware
  compatibility notes.
- **Port already in use** — change the host port with `PORT=<port> bash docker/run.sh`.
- For other environment questions, see [FAQ.md](../FAQ.md).
