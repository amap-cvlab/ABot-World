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
FP8 quantization kernels with automatic backend selection.

This module automatically selects between Triton (for Linux/CUDA) and
PyTorch (for Windows/CPU) implementations based on the runtime environment.
"""

# from angelslim.compressor._platform import use_triton

# Conditional imports based on platform/backend availability
# if use_triton():
from .fp8_per_block import fp8_per_block_quant_triton
from .fp8_per_token_group import fp8_per_token_group_quant_triton
# else:
#     # PyTorch fallback implementations
#     from .fp8_per_block_torch import (
#         fp8_per_block_quant_torch as fp8_per_block_quant_triton,
#     )
#     from .fp8_per_token_group_torch import (
#         fp8_per_token_group_quant_torch as fp8_per_token_group_quant_triton,
#     )

# Also export PyTorch versions directly for explicit use
from .fp8_per_block_torch import fp8_per_block_quant_torch
from .fp8_per_token_group_torch import fp8_per_token_group_quant_torch

__all__ = [
    "fp8_per_token_group_quant_triton",
    "fp8_per_block_quant_triton",
    "fp8_per_block_quant_torch",
    "fp8_per_token_group_quant_torch",
]
