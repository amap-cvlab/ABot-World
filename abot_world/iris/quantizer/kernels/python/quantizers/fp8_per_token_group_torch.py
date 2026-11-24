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
Pure PyTorch implementation of per-token-group FP8 quantization.

This module provides CPU/Windows-compatible implementations that mirror
the Triton kernels for FP8 per-token-group quantization.
"""

from typing import Tuple

import torch


def fp8_per_token_group_quant_torch(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of per-token-group FP8 quantization.

    Function to perform per-token-group quantization on an input tensor `x`.
    Each group of `group_size` elements along the last dimension is quantized
    with its own scale factor.

    Args:
        x: Input tensor, must be contiguous
        group_size: Size of each quantization group
        eps: Small value to avoid division by zero (default: 1e-10)
        dtype: Target FP8 dtype (default: float8_e4m3fn)
        column_major_scales: Whether to use column-major scale layout (default: False)
        scale_tma_aligned: Whether to use TMA-aligned scales (default: False)

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    assert (
        x.shape[-1] % group_size == 0
    ), f"the last dimension ({x.shape[-1]}) cannot be divisible by group_size ({group_size})"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    device = x.device
    orig_shape = x.shape
    num_groups = orig_shape[-1] // group_size

    # Reshape for group-wise processing: [..., N] -> [..., num_groups, group_size]
    x_grouped = x.view(*orig_shape[:-1], num_groups, group_size)

    # Convert to float32 for computation
    x_float = x_grouped.to(torch.float32)

    # Compute per-group max absolute values
    absmax = x_float.abs().amax(dim=-1, keepdim=True)  # [..., num_groups, 1]
    absmax = torch.maximum(absmax, torch.tensor(eps, device=device, dtype=torch.float32))

    # Compute scales
    scales = absmax / fp8_max  # [..., num_groups, 1]

    # Quantize
    x_scaled = x_float / scales
    x_q = x_scaled.clamp(fp8_min, fp8_max).to(dtype)

    # Reshape back to original shape
    x_q = x_q.view(orig_shape)

    # Prepare scale tensor based on output format
    scales = scales.squeeze(-1)  # [..., num_groups]

    if column_major_scales:
        if scale_tma_aligned:
            # TMA-aligned column-major scales
            aligned_size = (orig_shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                orig_shape[:-2] + (num_groups, aligned_size),
                device=device,
                dtype=torch.float32,
            )
            # Transpose and fill
            scales_transposed = scales.permute(*range(len(orig_shape) - 2), -1, -2).contiguous()
            x_s[..., : orig_shape[-2]] = scales_transposed
            x_s = x_s.permute(-1, -2)[: orig_shape[-2], :]
        else:
            # Standard column-major scales
            x_s = scales.permute(*range(len(orig_shape) - 2), -1, -2).contiguous()
    else:
        # Row-major scales (default)
        x_s = scales

    return x_q, x_s.float()


def fp8_per_token_group_quant_torch_simple(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified PyTorch implementation for row-major scales only.

    This version is more straightforward and easier to understand.

    Args:
        x: Input tensor, must be contiguous
        group_size: Size of each quantization group
        eps: Small value to avoid division by zero
        dtype: Target FP8 dtype

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    finfo = torch.finfo(dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    orig_shape = x.shape
    num_groups = orig_shape[-1] // group_size

    # Flatten all dimensions except last, then reshape for groups
    x_flat = x.view(-1, orig_shape[-1])  # [M, N]
    M = x_flat.shape[0]

    # Reshape to [M, num_groups, group_size]
    x_grouped = x_flat.view(M, num_groups, group_size).float()

    # Compute per-group absmax
    absmax = x_grouped.abs().amax(dim=-1, keepdim=True)  # [M, num_groups, 1]
    absmax = torch.clamp(absmax, min=eps)

    # Compute scales
    scales = absmax / fp8_max

    # Quantize
    x_q = (x_grouped / scales).clamp(fp8_min, fp8_max).to(dtype)

    # Reshape back
    x_q = x_q.view(orig_shape)
    scales = scales.squeeze(-1).view(*orig_shape[:-1], num_groups)

    return x_q, scales.float()
