# Frequently Asked Questions (FAQ)

This document addresses common questions regarding hardware requirements, environment setup, and runtime compatibility for **ABot-World**.

---

## Hardware Compatibility

### Can ABot-World run on NVIDIA GeForce RTX 3090 / 4090?

**Short answer:** In principle, yes for GPUs with sufficient VRAM; RTX 4090 has been verified in our tests, while RTX 3090 has not yet been evaluated on our side.

**Details.** Peak GPU memory usage is approximately **19 GB**, and we recommend at least **64 GB** of system RAM. Under these constraints, consumer GPUs such as the RTX 3090 (24 GB) and RTX 4090 (24 GB) are theoretically capable of running the released causal student model.

We have successfully run inference on **RTX 4090**. We currently do not have access to an RTX 3090 for internal validation; therefore, RTX 3090 support remains unverified, though we expect it to be feasible given the memory footprint. Users attempting deployment on 3090/4090-class hardware should ensure adequate host memory and a CUDA environment consistent with the [Setup](README.md#-setup) instructions (CUDA 12.8, PyTorch 2.8, etc.).

---

## Related Resources

- Installation and environment setup: [README.md](README.md#-setup)
- FlashAttention glibc issues: [Dao-AILab/flash-attention#1708](https://github.com/Dao-AILab/flash-attention/issues/1708)
- SageAttention source build: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/main)
