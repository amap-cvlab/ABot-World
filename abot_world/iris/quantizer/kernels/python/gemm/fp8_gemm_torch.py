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
Pure PyTorch implementation of FP8 block-wise GEMM.

This module provides CPU/Windows-compatible implementations that mirror
the Triton kernel for FP8 GEMM with block-wise quantization.
"""

from typing import Optional

import torch


def fp8_gemm_torch_block(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of FP8 GEMM with block-wise quantization.

    Performs a matrix multiplication using FP8 precision with per-block scaling.
    This implementation dequantizes the inputs and performs standard matmul.

    C = (A * A_scale) @ (B * B_scale).T + bias

    Args:
        a: Input activation tensor in FP8 format, shape [..., K]
        a_s: Scale tensor for A, shape [..., K // block_size] or [..., num_k_blocks]
        b: Weight tensor in FP8 format, shape [N, K]
        b_s: Scale tensor for B, shape [N // block_size, K // block_size]
        out_dtype: Output data type (default: bfloat16)
        bias: Optional bias tensor, shape [N]
        block_size: Block size used for quantization (default: 128)

    Returns:
        Output tensor of shape [..., N]
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    K = a.size(-1)
    orig_shape = a.shape[:-1]
    M = a.numel() // K
    N = b.size(0)

    # Reshape for computation
    a_2d = a.view(M, K)  # [M, K]

    # Dequantize A: expand scales to match tensor dimensions
    # a_s shape is typically [M, K//block_size]
    a_s_2d = a_s.view(M, -1)  # [M, num_k_blocks]

    # Dequantize by expanding scales
    a_dq = _dequantize_per_group(a_2d, a_s_2d, block_size, K)

    # Dequantize B: b_s is [N//block_size, K//block_size]
    b_dq = _dequantize_blockwise_2d(b, b_s, block_size)

    # Perform matmul: [M, K] @ [K, N] -> [M, N]
    c = torch.matmul(a_dq.to(out_dtype), b_dq.to(out_dtype).t())

    # Reshape output
    c = c.view(*orig_shape, N)

    if bias is not None:
        c = c + bias

    return c


def _dequantize_per_group(
    x: torch.Tensor,
    s: torch.Tensor,
    group_size: int,
    K: int,
) -> torch.Tensor:
    """
    Dequantize tensor with per-group scales.

    Args:
        x: Quantized tensor [M, K]
        s: Scale tensor [M, num_groups]
        group_size: Size of each group
        K: Total size of last dimension

    Returns:
        Dequantized tensor [M, K]
    """
    M = x.shape[0]
    num_groups = s.shape[1]

    x_float = x.to(torch.float32)

    # Expand scales to match K dimension
    # s: [M, num_groups] -> [M, K]
    s_expanded = s.unsqueeze(-1).expand(M, num_groups, group_size)
    s_expanded = s_expanded.reshape(M, num_groups * group_size)

    # Handle case where K is not exactly num_groups * group_size
    if s_expanded.shape[1] > K:
        s_expanded = s_expanded[:, :K]
    elif s_expanded.shape[1] < K:
        # Pad with last scale value
        pad_size = K - s_expanded.shape[1]
        s_expanded = torch.nn.functional.pad(s_expanded, (0, pad_size), mode="replicate")

    return x_float * s_expanded


def _dequantize_blockwise_2d(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Dequantize 2D tensor with block-wise scales.

    Args:
        x: Quantized tensor [N, K]
        s: Scale tensor [n_blocks, k_blocks]
        block_size: Block size

    Returns:
        Dequantized tensor [N, K]
    """
    N, K = x.shape
    n_blocks, k_blocks = s.shape

    x_float = x.to(torch.float32)
    y = torch.empty_like(x_float)

    for nb in range(n_blocks):
        n_start = nb * block_size
        n_end = min(n_start + block_size, N)
        for kb in range(k_blocks):
            k_start = kb * block_size
            k_end = min(k_start + block_size, K)

            scale = s[nb, kb]
            y[n_start:n_end, k_start:k_end] = x_float[n_start:n_end, k_start:k_end] * scale

    return y


def fp8_gemm_torch_simple(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simplified PyTorch FP8 GEMM using full dequantization.

    This version is simpler but may use more memory for large tensors.

    Args:
        a: Input activation tensor in FP8 format
        a_s: Scale tensor for A
        b: Weight tensor in FP8 format
        b_s: Scale tensor for B
        out_dtype: Output data type
        bias: Optional bias tensor

    Returns:
        Output tensor
    """
    K = a.size(-1)
    orig_shape = a.shape[:-1]
    M = a.numel() // K
    N = b.size(0)

    # Reshape
    a_2d = a.view(M, K)
    a_s_2d = a_s.view(M, -1)

    # Simple dequantization: repeat scales to match dimensions
    block_size = K // a_s_2d.shape[1] if a_s_2d.shape[1] > 0 else K

    # Dequantize A
    a_dq = a_2d.to(torch.float32)
    if a_s_2d.shape[1] > 1:
        a_s_expanded = a_s_2d.repeat_interleave(block_size, dim=1)[:, :K]
        a_dq = a_dq * a_s_expanded
    else:
        a_dq = a_dq * a_s_2d

    # Dequantize B
    b_dq = _dequantize_blockwise_2d(b, b_s, block_size)

    # Matmul
    c = torch.matmul(a_dq.to(out_dtype), b_dq.to(out_dtype).t())
    c = c.view(*orig_shape, N)

    if bias is not None:
        c = c + bias

    return c
