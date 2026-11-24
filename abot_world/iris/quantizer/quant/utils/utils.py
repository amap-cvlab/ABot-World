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


import gc
import re
from typing import List, Optional, Union

import torch

__all__ = [
    "replace_module",
    "cleanup_memory",
    "should_quantize_layer",
    "_compile_pattern",
    "_ensure_deep_gemm",
    "_ensure_sgl_kernel",
    "QuantType",
]


class QuantType:
    FP8_PER_TENSOR = "fp8-per-tensor"
    FP8_PER_TOKEN = "fp8-per-token"
    FP8_PER_BLOCK = "fp8-per-block"
    FP8_PER_CHANNEL_VLLM = "fp8-per-channel-vllm"
    FP8_PER_TENSOR_WEIGHT_ONLY = "fp8-per-tensor-weight-only"
    FP8_PER_TOKEN_SGL = "fp8-per-token-sgl"
    INT8 = "int8"
    INT8_TORCHAO = "int8-torchao"
    INT8_VLLM = "int8-vllm"
    INT8_TRITON = "int8-triton"
    INT8_SGL = "int8-sgl"
    INT8_Q8F = "int8-q8f"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"
    MXFP6 = "mxfp6"
    MXFP8 = "mxfp8"
    VALID_TYPES = [
        FP8_PER_TENSOR,
        FP8_PER_TOKEN,
        FP8_PER_BLOCK,
        FP8_PER_CHANNEL_VLLM,
        FP8_PER_TENSOR_WEIGHT_ONLY,
        FP8_PER_TOKEN_SGL,
        INT8,
        INT8_TORCHAO,
        INT8_VLLM,
        INT8_TRITON,
        INT8_SGL,
        INT8_Q8F,
        NVFP4,
        MXFP4,
        MXFP6,
        MXFP8,
    ]

    @classmethod
    def validate(cls, quant_type: str):
        if quant_type not in cls.VALID_TYPES:
            raise ValueError(f"Invalid quant_type: {quant_type}. Valid types: {cls.VALID_TYPES}")


def replace_module(model: torch.nn.Module, name: str, new_module: torch.nn.Module):
    """
    Replace a submodule in the model with a new module by name.
    """
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, new_module)


def cleanup_memory():
    """
    Run garbage collection and clear CUDA memory cache.
    """
    gc.collect()
    torch.cuda.empty_cache()


def _compile_pattern(pattern: Union[str, re.Pattern], case_sensitive: bool = False) -> re.Pattern:
    """
    Compile a pattern (string or pre-compiled pattern) into a regex pattern object.

    Args:
        pattern: String pattern or already-compiled regex pattern.
        case_sensitive: Whether the match is case sensitive.

    Returns:
        Compiled regex pattern object.
    """
    if isinstance(pattern, str):
        # If the string contains special regex characters, treat as regex.
        if any(char in pattern for char in ".*+?^${}[]|()\\"):
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.compile(pattern, flags)
        else:
            # Escape regular string so it matches literally.
            escaped = re.escape(pattern)
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.compile(escaped, flags)
    else:
        # Already a compiled pattern.
        return pattern


def should_quantize_layer(
    layer_name: str,
    include_patterns: Optional[List[Union[str, re.Pattern]]] = None,
    exclude_patterns: Optional[List[Union[str, re.Pattern]]] = None,
    case_sensitive: bool = False,
) -> bool:
    """
    Decide whether a layer should be quantized based on inclusion/exclusion patterns.

    Args:
        layer_name: Name of the layer.
        include_patterns: List of patterns (str or regex) to include.
        exclude_patterns: List of patterns (str or regex) to exclude.
        case_sensitive: Whether patterns are matched case sensitively.

    Returns:
        bool: Whether this layer should be quantized.

    Note:
        String patterns are auto-detected as regex if they include special chars:
        - If contains any . * + ? ^ $ { } [ ] | ( ) \\ -> treated as regex
        - Otherwise, escaped and matched literally
    """
    if include_patterns is None:
        include_patterns = []
    if exclude_patterns is None:
        exclude_patterns = []

    # Check exclusion patterns
    for pattern in exclude_patterns:
        compiled_pattern = _compile_pattern(pattern, case_sensitive)
        if compiled_pattern.search(layer_name):
            return False

    # If no include patterns, default is to include all layers
    if not include_patterns:
        return True

    # Check inclusion patterns
    for pattern in include_patterns:
        compiled_pattern = _compile_pattern(pattern, case_sensitive)
        if compiled_pattern.search(layer_name):
            return True

    return False


_deep_gemm_cached = None


def _ensure_deep_gemm():
    """
    Lazy, safe import of deep_gemm with process-level caching. Returns the module
    if available, otherwise raises a clear error.
    """
    global _deep_gemm_cached
    if _deep_gemm_cached is not None:
        return _deep_gemm_cached
    try:
        import deep_gemm

        _deep_gemm_cached = deep_gemm
        return _deep_gemm_cached
    except ImportError as e:
        raise ImportError(
            (
                "deep_gemm is required for 'fp8-per-block' quantization with "
                "native_fp8_support, but was not found. Please install deep_gemm first."
            )
        ) from e


_sgl_kernel_cached = None


def _ensure_sgl_kernel():
    """
    Lazy, safe import of sgl_kernel with process-level caching. Returns the module
    if available, otherwise raises a clear error.
    """
    global _sgl_kernel_cached
    if _sgl_kernel_cached is not None:
        return _sgl_kernel_cached
    try:
        import sgl_kernel

        _sgl_kernel_cached = sgl_kernel
        return _sgl_kernel_cached
    except ImportError as e:
        raise ImportError(
            (
                "sgl_kernel is required for 'fp8-per-token-sgl' quantization with "
                "native_fp8_support, but was not found. Please install sgl_kernel first"
            )
        ) from e
