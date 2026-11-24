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

"""
Pure PyTorch implementation of per-block FP8 quantization.

This module provides CPU/Windows-compatible implementations that mirror
the Triton kernel for FP8 per-block quantization.
"""

from typing import Tuple

import torch

# FP8 E4M3 max value
FP8_MAX = 448.0


def fp8_per_block_quant_torch(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of block-wise FP8 quantization.

    Quantizes a FP32 2D tensor to FP8 (E4M3FN) using block-wise quantization.
    For each (block_size x block_size) block:
        - scale = max(abs(block)) / 448.0 (FP8 E4M3FN max magnitude)
        - if block is all zeros, use scale = 1.0 to avoid div-by-zero
        - scale, clamp and cast to FP8

    Args:
        x: Input tensor of shape (M, N), must be contiguous
        block_size: Size of each quantization block (default: 128)

    Returns:
        Tuple of (quantized_tensor, scale_tensor):
            - y: Quantized FP8 tensor, same shape as input
            - s: Per-block scales, shape (num_blocks_M, num_blocks_N)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must be 2D"

    M, N = x.size()
    device = x.device

    # Calculate number of blocks
    m_blocks = (M + block_size - 1) // block_size
    n_blocks = (N + block_size - 1) // block_size

    # Output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((m_blocks, n_blocks), dtype=torch.float32, device=device)

    # Convert to float32 for computation
    x_float = x.to(torch.float32)

    # Process each block
    for mb in range(m_blocks):
        m_start = mb * block_size
        m_end = min(m_start + block_size, M)
        for nb in range(n_blocks):
            n_start = nb * block_size
            n_end = min(n_start + block_size, N)

            block = x_float[m_start:m_end, n_start:n_end]
            max_val = block.abs().amax()

            # Compute scale (guard against zero)
            scale = max_val / FP8_MAX
            if scale.item() == 0.0:
                scale = torch.tensor(1.0, dtype=torch.float32, device=device)

            # Quantize block
            y_block = (block / scale).to(torch.float8_e4m3fn)

            # Store results
            y[m_start:m_end, n_start:n_end] = y_block
            s[mb, nb] = scale

    return y, s


def fp8_per_block_quant_torch_fast(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized PyTorch implementation using tensor operations.

    This is faster than the loop-based version for large tensors on GPU.

    Args:
        x: Input tensor of shape (M, N), must be contiguous
        block_size: Size of each quantization block (default: 128)

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must be 2D"

    M, N = x.size()

    # Pad tensor to be divisible by block_size
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size

    if pad_m > 0 or pad_n > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad_n, 0, pad_m), value=0.0)
    else:
        x_padded = x

    M_padded, N_padded = x_padded.size()
    m_blocks = M_padded // block_size
    n_blocks = N_padded // block_size

    # Reshape to blocks: [m_blocks, block_size, n_blocks, block_size]
    x_blocks = x_padded.view(m_blocks, block_size, n_blocks, block_size)
    x_blocks = x_blocks.permute(0, 2, 1, 3).contiguous()

    # Compute max absolute value per block
    x_float = x_blocks.to(torch.float32)
    max_vals = x_float.abs().amax(dim=(2, 3))  # [m_blocks, n_blocks]

    # Compute scales
    s = max_vals / FP8_MAX
    s = torch.where(s == 0.0, torch.ones_like(s), s)

    # Quantize
    s_expanded = s[:, :, None, None]
    y_blocks = (x_float / s_expanded).to(torch.float8_e4m3fn)

    # Reshape back
    y_blocks = y_blocks.permute(0, 2, 1, 3).contiguous()
    y_padded = y_blocks.view(M_padded, N_padded)

    # Remove padding
    y = y_padded[:M, :N].contiguous()

    # Adjust scale tensor size if needed
    actual_m_blocks = (M + block_size - 1) // block_size
    actual_n_blocks = (N + block_size - 1) // block_size
    s = s[:actual_m_blocks, :actual_n_blocks].contiguous()

    return y, s
