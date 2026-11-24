# Frequently Asked Questions (FAQ)

This document addresses common questions regarding hardware requirements, software environment, and runtime compatibility for **ABot-World**.

---

## Hardware Compatibility

### Can ABot-World run on NVIDIA GeForce RTX 3090 / 4090?

GPU memory depends on the decoder, resolution and precision. The default
v0.5 BF16 configuration does not inherit the v0 19 GB memory measurement.
Native FP8 per-token requires compute capability 8.9 or newer. See
[Setup](README.md#setup).

### Is multi-GPU inference supported?

Not at present. The current release targets **single-GPU** desktop inference. Multi-GPU parallelism could, in principle, improve throughput, but has not been systematically validated. We recommend single-GPU deployment until multi-GPU support is officially released.

---

## Software Environment

### Which CUDA versions are supported?

Use the CUDA environment described in the [Setup](README.md#setup) section.

### Is Windows supported? Which operating system is recommended?

We strongly recommend **Linux**. Support for certain components under Windows is limited or fragile. For reproducible installation and inference, please prefer Linux (e.g., Ubuntu 22.04 as in our reference setup).

---

## Gradio Demo

### Which browser should I use for the Gradio interface?

We recommend **Google Chrome** for online Gradio usage, where we have observed the most consistent behavior.

---

## Related Resources

- Installation and environment setup: [README.md](README.md#setup)
- FlashAttention glibc issues: [Dao-AILab/flash-attention#1708](https://github.com/Dao-AILab/flash-attention/issues/1708)
- SageAttention source build: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/main)
