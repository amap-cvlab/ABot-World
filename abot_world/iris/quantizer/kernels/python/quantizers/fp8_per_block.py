# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import triton
import triton.language as tl


# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def _fp8_per_block_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Quantizes FP32 tensor to FP8 format using block-wise quantization."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    max_val = tl.max(tl.abs(x))
    scale = max_val / 448.0
    scale = tl.where(max_val == 0.0, 1.0, scale)
    y = x / scale
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, scale)


# triton implementation
# for weight quantization on gpu
def fp8_per_block_quant_triton(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a FP32 2D tensor to FP8 (E4M3FN) using block-wise quantization.
    For each (block_size x block_size) block:
        - scale = max(abs(block)) / 448.0 (FP8 E4M3FN max magnitude)
        - if block is all zeros, use scale = 1.0 to avoid div-by-zero
        - scale, clamp and cast to FP8
    Returns:
        y: Quantized FP8 tensor, same shape as input
        s: Per-block scales, shape (num_blocks_M, num_blocks_N)
    """
    assert x.is_contiguous()
    assert x.dim() == 2

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    m_blocks = triton.cdiv(M, block_size)
    n_blocks = triton.cdiv(N, block_size)
    s = torch.empty((m_blocks, n_blocks), dtype=torch.float32, device=x.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(N, meta["BLOCK_SIZE"]),
        )

    _fp8_per_block_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)

    return y, s
