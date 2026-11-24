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

from typing import Dict, List, Optional

from .quant import *  # noqa: F401 F403
from .quant import DynamicDiTQuantizer

DEFAULT_FP8_INCLUDE_PATTERNS = ["blocks"]
DEFAULT_FP8_EXCLUDE_PATTERNS = []


def apply_fp8_quantization(
    model,
    quant_type: str = "fp8-per-token",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Apply DynamicDiTQuantizer to the provided DiT model."""
    final_include_patterns = (
        include_patterns if include_patterns is not None else DEFAULT_FP8_INCLUDE_PATTERNS
    )
    final_exclude_patterns = (
        exclude_patterns if exclude_patterns is not None else DEFAULT_FP8_EXCLUDE_PATTERNS
    )

    quantizer = DynamicDiTQuantizer(
        quant_type=quant_type,
        include_patterns=final_include_patterns,
        exclude_patterns=final_exclude_patterns,
    )
    quantizer.convert_linear(model)
    return {
        "quantizer": quantizer,
        "quant_type": quant_type,
        "include_patterns": final_include_patterns,
        "exclude_patterns": final_exclude_patterns,
    }
