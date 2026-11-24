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
FP8 GEMM kernels with automatic backend selection.

This module automatically selects between Triton (for Linux/CUDA) and
PyTorch (for Windows/CPU) implementations based on the runtime environment.
"""

# from angelslim.compressor._platform import use_triton

# Conditional imports based on platform/backend availability
# if use_triton():
from .fp8_gemm import fp8_gemm_triton_block
# else:
#     # PyTorch fallback implementation
#     from .fp8_gemm_torch import fp8_gemm_torch_block as fp8_gemm_triton_block

# Also export PyTorch version directly for explicit use
from .fp8_gemm_torch import fp8_gemm_torch_block

__all__ = ["fp8_gemm_triton_block", "fp8_gemm_torch_block"]
