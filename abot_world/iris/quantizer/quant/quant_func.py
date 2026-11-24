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

from typing import Optional, Tuple

import torch

from ..kernels.python.gemm import fp8_gemm_triton_block
from lightx2v_kernel.gemm import (
    cutlass_scaled_mxfp4_mm,
    cutlass_scaled_mxfp6_mxfp8_mm,
    cutlass_scaled_mxfp8_mm,
    cutlass_scaled_nvfp4_mm,
    scaled_mxfp4_quant,
    scaled_mxfp6_quant,
    scaled_mxfp8_quant,
    scaled_nvfp4_quant,
)
from ..kernels.python.quantizers import (
    fp8_per_block_quant_triton,
    fp8_per_token_group_quant_triton,
)
from .utils import QuantType, _ensure_deep_gemm, _ensure_sgl_kernel

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)


# quant function for per-tensor fp8
# modified from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
def fp8_per_tensor_quant(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Dynamically Quantize a tensor using per-tensor static scaling factor.
    Args:
        x: The input tensor.
    """
    if x.numel() == 0:
        # handle empty tensor for empty MoE experts
        min_val, max_val = (
            torch.tensor(-16.0, dtype=x.dtype),
            torch.tensor(16.0, dtype=x.dtype),
        )
    else:
        min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = FP8_MAX / amax.clamp(min=1e-12)
    qx = (x * scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qx = qx.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qx, scale


# quant function for per-token and per-group fp8
def fp8_per_token_group_quant(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token-group FP8 quantization with automatic backend selection.

    Uses Triton kernel on GPU when available, otherwise falls back to PyTorch.

    Args:
        x: Input tensor
        group_size: Size of each quantization group
        eps: Small value to avoid division by zero
        dtype: Target FP8 data type
        column_major_scales: Whether to use column-major scale layout
        scale_tma_aligned: Whether to use TMA-aligned scales

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    # fp8_per_token_group_quant_triton is automatically selected based on
    # backend availability (Triton vs PyTorch) via __init__.py
    return fp8_per_token_group_quant_triton(
        x,
        group_size,
        eps,
        dtype,
        column_major_scales,
        scale_tma_aligned,
    )


def fp8_per_channel_quant(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel FP8 weight quantization (E4M3 format)

    Args:
        weight: Original weight tensor with shape [out_features, in_features]

    Returns:
        weight_quant: Quantized weight [out_features, in_features], dtype=float8_e4m3fn
        weight_scale: Scale factors [out_features, 1], dtype=float32
    """
    abs_max = torch.abs(weight).amax(dim=1, keepdim=True)  # [out_features, 1]

    weight_scale = abs_max / FP8_MAX
    weight_scale = torch.clamp(weight_scale, min=1e-12)

    weight_scaled = (weight / weight_scale).clamp(min=FP8_MIN, max=FP8_MAX)
    weight_quant = weight_scaled.to(torch.float8_e4m3fn)

    return weight_quant, weight_scale.float()


def int8_per_channel_quant(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel symmetric INT8 quantization for linear weights.

    Args:
        weight: Weight tensor with shape [out_features, in_features]

    Returns:
        weight_quant: Quantized INT8 weight [out_features, in_features]
        weight_scale: Per-channel scales [out_features, 1], float32
    """
    abs_max = torch.abs(weight).amax(dim=1, keepdim=True).clamp(min=1e-5)
    qmin, qmax = -128, 127
    weight_scale = (abs_max / qmax).to(torch.float32)
    weight_quant = torch.clamp(torch.round(weight / weight_scale), qmin, qmax).to(torch.int8)
    return weight_quant, weight_scale


def nvfp4_per_tensor_quant(
    x: torch.Tensor, input_global_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor NVFP4 quantization using LightX2V kernel.

    Returns packed uint8 values and per-block quantized scales.
    """
    if x.dim() != 2:
        raise ValueError(f"nvfp4_per_tensor_quant expects 2D tensor, but got shape={tuple(x.shape)}")

    if input_global_scale is None:
        x_absmax = torch.abs(x).amax().float().clamp(min=1e-12)
        input_global_scale = (2688.0 / x_absmax).to(torch.float32)
    else:
        input_global_scale = input_global_scale.to(device=x.device, dtype=torch.float32)

    qx, qscale = scaled_nvfp4_quant(x, input_global_scale)
    return qx, qscale, input_global_scale


def mxfp4_per_tensor_quant(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor MXFP4 quantization using LightX2V kernel.

    Returns packed uint8 values and per-block quantized scales.
    """
    if x.dim() != 2:
        raise ValueError(f"mxfp4_per_tensor_quant expects 2D tensor, but got shape={tuple(x.shape)}")

    qx, qscale = scaled_mxfp4_quant(x)
    # Keep a compatible triplet return signature with NVFP4 paths.
    input_global_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    return qx, qscale, input_global_scale


def mxfp6_per_tensor_quant(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor MXFP6 quantization using LightX2V kernel.

    Returns packed uint8 values and per-block quantized scales.
    """
    if x.dim() != 2:
        raise ValueError(f"mxfp6_per_tensor_quant expects 2D tensor, but got shape={tuple(x.shape)}")

    qx, qscale = scaled_mxfp6_quant(x)
    input_global_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    return qx, qscale, input_global_scale


def mxfp8_per_tensor_quant(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor MXFP8 quantization using LightX2V kernel.

    Returns packed uint8 values and per-block quantized scales.
    """
    if x.dim() != 2:
        raise ValueError(f"mxfp8_per_tensor_quant expects 2D tensor, but got shape={tuple(x.shape)}")

    qx, qscale = scaled_mxfp8_quant(x)
    input_global_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    return qx, qscale, input_global_scale


def fp8_per_token_quant_sgl(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    input_tensor_quant = torch.empty(
        (m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False
    )
    input_tensor_scale = torch.empty(
        (m, 1), dtype=torch.float32, device="cuda", requires_grad=False
    )
    _sgl_kernel = _ensure_sgl_kernel()
    _sgl_kernel.sgl_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
    return input_tensor_quant, input_tensor_scale


# pure torch implementation of block-wise FP8 quantization on cpu
def fp8_per_block_quant_torch(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure Torch implementation of block-wise FP8 (e4m3fn) quantization.

    Args:
        x (torch.Tensor): 2D tensor (M, N) in float types.
        block_size (int): Block size for both M and N dimensions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - y: quantized tensor with dtype=torch.float8_e4m3fn and same
              shape/device as x
            - s: per-block scale tensor with shape (ceil(M/bs), ceil(N/bs)) on
              x.device
    """
    assert x.is_contiguous()
    assert x.dim() == 2

    M, N = x.size()
    device = x.device

    # Output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    m_blocks = (M + block_size - 1) // block_size
    n_blocks = (N + block_size - 1) // block_size
    s = torch.empty((m_blocks, n_blocks), dtype=torch.float32, device=device)

    # Iterate over blocks
    for mb in range(m_blocks):
        m_start = mb * block_size
        m_end = min(m_start + block_size, M)
        for nb in range(n_blocks):
            n_start = nb * block_size
            n_end = min(n_start + block_size, N)

            x_block = x[m_start:m_end, n_start:n_end].to(torch.float32)
            # Compute per-block scale: max(abs(x))/448.0, guard zero to 1.0
            max_val = x_block.abs().amax().to(torch.float32)
            scale = max_val / FP8_MAX
            if scale.item() == 0.0:
                scale = torch.tensor(1.0, dtype=torch.float32, device=device)

            # Quantize block
            y_block = (x_block / scale).to(torch.float8_e4m3fn)

            # Store results
            y[m_start:m_end, n_start:n_end] = y_block
            s[mb, nb] = scale

    return y, s


# quant function for per-block fp8
def fp8_per_block_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block FP8 quantization with automatic GPU/CPU dispatch.

    Quantizes a FP32 2D tensor to FP8 (E4M3FN) using block-wise quantization.
    For each (block_size x block_size) block:
        - scale = max(abs(block)) / 448.0 (FP8 E4M3FN max magnitude)
        - if block is all zeros, use scale = 1.0 to avoid div-by-zero
        - scale, clamp and cast to FP8

    Args:
        x: Input tensor (2D)
        block_size: Block size for quantization

    Returns:
        Tuple of (quantized_tensor, scale_tensor):
            - y: Quantized FP8 tensor, same shape as input
            - s: Per-block scales, shape (num_blocks_M, num_blocks_N)
    """
    if x.is_cuda:
        return fp8_per_block_quant_triton(x, block_size)
    else:
        return fp8_per_block_quant_torch(x, block_size)


# gemm function for per-block quantization fp8 using deepgemm
def fp8_gemm_deepgemm_block(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    origin_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    a_fp8 = (a, a_s)
    b_fp8 = (b, b_s)
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)

    _deep_gemm = _ensure_deep_gemm()
    _deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, out)

    if origin_shape is not None:
        out = out.reshape([*origin_shape[:-1], b.shape[0]])
    if bias is not None:
        out += bias

    return out.to(out_dtype)


# gemm function for per-token and per-group fp8 quantization using torch native api
def fp8_gemm_torch_tensor_token(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    need_reshape = a.dim() == 3
    if need_reshape:
        batch_size = a.shape[0]
        A_input = a.reshape(-1, a.shape[-1])
    else:
        batch_size = None
        A_input = a

    output = torch._scaled_mm(
        A_input,
        b.t(),
        out_dtype=out_dtype,
        scale_a=a_s,
        scale_b=b_s,
        bias=bias,
    )
    # If output is a tuple, take the first element
    if isinstance(output, tuple):
        output = output[0]

    if need_reshape:
        output = output.reshape(batch_size, output.shape[0] // batch_size, output.shape[1])

    return output


def fp8_weight_only_gemm(A, B, B_scale, bias, out_dtype):
    """Perform FP8 GEMM operation with fallback to standard linear.

    Args:
        A: Input tensor A.
        B: Input tensor B.
        B_scale: Scale factor for tensor B.
        bias: Optional bias tensor.
        out_dtype: Output data type.
        native_fp8_support: Whether to use native FP8 support.
        quant_type: Quantization type.
        origin_shape: Original shape for reshaping.

    Returns:
        torch.Tensor: Result of the GEMM operation.
    """
    if A.numel() == 0:
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    output = torch.nn.functional.linear(
        A.to(out_dtype),
        B.to(out_dtype) * B_scale.to(out_dtype),
        bias=bias,
    )

    return output


def fp8_gemm_sgl_token(A, A_scale, B, B_scale, out_dtype, bias):
    """GEMM function for FP8 per-token-sgl quantization using sgl-kernel.

    Args:
        A: Input activation tensor
        A_scale: Scale tensor for input activations
        B: Weight tensor
        B_scale: Scale tensor for weights
        out_dtype: Output data type.
        bias: Optional bias tensor

    Returns:
        torch.Tensor: Result of the GEMM operation.
    """
    _sgl_kernel = _ensure_sgl_kernel()
    shape = (A.shape[0], B.shape[0])
    output = torch.empty(shape, dtype=out_dtype, device=A.device, requires_grad=False)
    output = _sgl_kernel.fp8_scaled_mm(
        A,
        B.t(),
        A_scale,
        B_scale.float(),
        out_dtype,
        bias=bias,
    )

    return output


def fp8_gemm(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    native_fp8_support: bool = False,
    quant_type: str = QuantType.FP8_PER_TENSOR,
    origin_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Unified GEMM function for FP8 quantization.

    Args:
        A: Input activation tensor
        A_scale: Scale tensor for input activations
        B: Weight tensor
        B_scale: Scale tensor for weights
        bias: Optional bias tensor
        out_dtype: Output data type
        native_fp8_support: Whether to use native FP8 kernels
        quant_type: Quantization type (QuantType enum values)
        origin_shape: Original shape for reshaping output

    Returns:
        Output tensor after GEMM operation

    Raises:
        ValueError: If unsupported combination of parameters is provided
    """
    # Return empty tensor if the input is empty
    if A.numel() == 0:
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    if quant_type == QuantType.FP8_PER_CHANNEL_VLLM:
        if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "cutlass_scaled_mm"):
            raise RuntimeError(
                "quant_type='fp8-per-channel-vllm' requires torch.ops._C.cutlass_scaled_mm"
            )
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
            requires_grad=False,
        )
        weight_scale = B_scale
        if (
            weight_scale.dim() == 2
            and weight_scale.shape[0] == B.shape[0]
            and weight_scale.shape[1] == 1
        ):
            weight_scale = weight_scale.t()
        torch.ops._C.cutlass_scaled_mm(
            output,
            A,
            B.t(),
            A_scale,
            weight_scale,
            bias,
        )
        return output

    # Prioritize native fp8 support if available
    if native_fp8_support:
        if quant_type in (QuantType.FP8_PER_TENSOR, QuantType.FP8_PER_TOKEN):
            # Use torch native fp8 GEMM for per-tensor and per-token fp8 quantization
            return fp8_gemm_torch_tensor_token(A, A_scale, B, B_scale, out_dtype, bias)
        elif quant_type == QuantType.FP8_PER_TOKEN_SGL:
            # Use sgl-kernel for per-token-sgl fp8 quantization
            return fp8_gemm_sgl_token(A, A_scale, B, B_scale, out_dtype, bias)
        elif quant_type == QuantType.FP8_PER_BLOCK:
            # Use deepgemm accelerated blockwise fp8 GEMM
            return fp8_gemm_deepgemm_block(A, A_scale, B, B_scale, out_dtype, bias, origin_shape)
    else:
        if quant_type == QuantType.FP8_PER_BLOCK:
            # Use triton kernel for blockwise fp8 quantization
            return fp8_gemm_triton_block(A, A_scale, B, B_scale, out_dtype, bias)
        elif quant_type == QuantType.FP8_PER_TENSOR:
            # Fall back to scaled linear for per-tensor fp8 without native fp8 kernel
            return torch.nn.functional.linear(
                A.to(out_dtype) * A_scale,
                B.to(out_dtype) * B_scale.to(out_dtype),
                bias=bias,
            )

    # Raise error for unsupported combination of quant_type and native_fp8_support
    raise ValueError(
        f"Unsupported combination: "
        f"\n  quant_type={quant_type},"
        f"\n  native_fp8_support={native_fp8_support}.\n"
        "Supported combinations:\n"
        "  - native_fp8_support=True, "
        "quant_type in [fp8-per-tensor, fp8-per-token,"
        " fp8-per-block, fp8-per-token-sgl]\n"
        "  - native_fp8_support=False, "
        "quant_type in [fp8-per-tensor, fp8-per-block]"
    )
