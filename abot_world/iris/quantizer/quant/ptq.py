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

import copy
import logging
import re
from typing import Callable, List, Optional, Tuple, Union

import torch
import tqdm

from .modules import FP4DynamicLinear, FP8DynamicLinear, FP8WeightOnlyLinear, INT8DynamicLinear
from .quant_func import (
    fp8_per_block_quant,
    fp8_per_channel_quant,
    fp8_per_tensor_quant,
    fp8_per_token_group_quant,
    int8_per_channel_quant,
    mxfp4_per_tensor_quant,
    mxfp6_per_tensor_quant,
    mxfp8_per_tensor_quant,
    nvfp4_per_tensor_quant,
)
from .utils import (
    QuantType,
    _ensure_deep_gemm,
    _ensure_sgl_kernel,
    cleanup_memory,
    load_fp8_scales,
    load_quantized_model,
    replace_module,
    save_quantized_model,
    should_quantize_layer,
)

__all__ = ["DynamicDiTQuantizer"]

logger = logging.getLogger(__name__)


class DynamicDiTQuantizer:
    """
    Quantizer for DiT that supports various FP8 quantization strategies.
    Optimized for efficient initialization and conversion.
    """

    def __init__(
        self,
        quant_type: str = QuantType.FP8_PER_TENSOR,
        layer_filter: Optional[Callable[[str], bool]] = None,
        include_patterns: Optional[List[Union[str, re.Pattern]]] = None,
        exclude_patterns: Optional[List[Union[str, re.Pattern]]] = None,
        native_fp8_support: Optional[bool] = None,
    ):
        QuantType.validate(quant_type)
        self.fp8_scales_map = {}
        self.quant_type = quant_type
        self.include_patterns = include_patterns or [
            "wrapped_module",
            "block",
            "lin",
            "img",
            "txt",
        ]
        self.exclude_patterns = exclude_patterns or ["embed"]

        # Configure layer filter function
        self.layer_filter = (
            layer_filter
            if layer_filter is not None
            else lambda name: should_quantize_layer(
                name, self.include_patterns, self.exclude_patterns
            )
        )

        # Auto-detect FP8 native support, fallback to False if not present
        if native_fp8_support is not None:
            self.native_fp8_support = native_fp8_support
        else:
            self.native_fp8_support = (
                torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
            )

        self.quantize_linear_module = self._set_quantize_linear_module()

    def _set_quantize_linear_module(self) -> torch.nn.Module:
        if self.quant_type in (
            QuantType.INT8,
            QuantType.INT8_TORCHAO,
            QuantType.INT8_VLLM,
            QuantType.INT8_TRITON,
            QuantType.INT8_SGL,
            QuantType.INT8_Q8F,
        ):
            return INT8DynamicLinear
        if self.quant_type in (QuantType.NVFP4, QuantType.MXFP4, QuantType.MXFP6, QuantType.MXFP8):
            return FP4DynamicLinear
        if "fp8" in self.quant_type:
            if self.quant_type == QuantType.FP8_PER_TENSOR_WEIGHT_ONLY:
                return FP8WeightOnlyLinear
            else:
                return FP8DynamicLinear
        raise ValueError(f"Invalid quant_type: {self.quant_type}")

    def _quantize_linear_weight(
        self, linear: torch.nn.Linear
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.quant_type == QuantType.FP8_PER_TENSOR:
            quant_weight, weight_scale = fp8_per_tensor_quant(linear.weight)
        elif self.quant_type == QuantType.FP8_PER_TOKEN:
            quant_weight, weight_scale = fp8_per_token_group_quant(
                linear.weight, linear.weight.shape[-1]
            )
            weight_scale = weight_scale.t()
        elif self.quant_type == QuantType.FP8_PER_TOKEN_SGL:
            if self.native_fp8_support:
                _ensure_sgl_kernel()
            quant_weight, weight_scale = fp8_per_channel_quant(linear.weight)
        elif self.quant_type == QuantType.FP8_PER_BLOCK:
            if self.native_fp8_support:
                _ensure_deep_gemm()
            quant_weight, weight_scale = fp8_per_block_quant(linear.weight)
        elif self.quant_type == QuantType.FP8_PER_CHANNEL_VLLM:
            quant_weight, weight_scale = fp8_per_channel_quant(linear.weight)
        elif self.quant_type == QuantType.FP8_PER_TENSOR_WEIGHT_ONLY:
            quant_weight, weight_scale = fp8_per_tensor_quant(linear.weight)
        elif self.quant_type in (
            QuantType.INT8,
            QuantType.INT8_TORCHAO,
            QuantType.INT8_VLLM,
            QuantType.INT8_TRITON,
            QuantType.INT8_SGL,
            QuantType.INT8_Q8F,
        ):
            quant_weight, weight_scale = int8_per_channel_quant(linear.weight)
        elif self.quant_type == QuantType.NVFP4:
            quant_weight, weight_scale, weight_global_scale = nvfp4_per_tensor_quant(linear.weight)
            return quant_weight, weight_scale, weight_global_scale
        elif self.quant_type == QuantType.MXFP4:
            quant_weight, weight_scale, weight_global_scale = mxfp4_per_tensor_quant(linear.weight)
            return quant_weight, weight_scale, weight_global_scale
        elif self.quant_type == QuantType.MXFP8:
            quant_weight, weight_scale, weight_global_scale = mxfp8_per_tensor_quant(linear.weight)
            return quant_weight, weight_scale, weight_global_scale
        elif self.quant_type == QuantType.MXFP6:
            # Mixed kernel route expects MXFP8-formatted weights.
            quant_weight, weight_scale, weight_global_scale = mxfp6_per_tensor_quant(linear.weight)
            return quant_weight, weight_scale, weight_global_scale
        else:
            raise ValueError(f"Invalid quant_type: {self.quant_type}")
        return quant_weight, weight_scale

    def _convert_linear_with_scale(
        self, model: torch.nn.Module, scale: Union[torch.Tensor, float]
    ):
        model.to(torch.bfloat16)
        assert scale is not None, "scale is required"
        self.fp8_scales_map = load_fp8_scales(scale)
        converted_count = 0
        for name, module in tqdm.tqdm(list(model.named_modules()), desc="converting linear"):
            if isinstance(module, torch.nn.Linear) and self.layer_filter(name):
                # Prefer $name.weight_scale, fallback to "$name" key if needed
                s = self.fp8_scales_map.get(f"{name}.weight_scale") or self.fp8_scales_map.get(
                    name
                )
                if s is None:
                    continue
                # import pdb; pdb.set_trace()
                weight = module.weight.to(torch.float8_e4m3fn)
                bias = copy.deepcopy(module.bias) if module.bias is not None else None
                quant_linear = self.quantize_linear_module(
                    weight=weight,
                    weight_scale=s.float(),
                    bias=bias,
                    native_fp8_support=self.native_fp8_support,
                    quant_type=self.quant_type,
                )
                replace_module(model, name, quant_linear)
                del module.weight, module.bias, module
                converted_count += 1
        print(f"[INIT] quantized linear layers: {converted_count}")
        cleanup_memory()

    def _convert_linear(self, model: torch.nn.Module):
        model.to(torch.bfloat16)
        named_modules = list(model.named_modules())
        converted_count = 0
        for name, module in tqdm.tqdm(named_modules, desc="converting linear"):
            if isinstance(module, torch.nn.Linear) and self.layer_filter(name):
                quantized = self._quantize_linear_weight(module)
                if self.quant_type in (
                    QuantType.NVFP4,
                    QuantType.MXFP4,
                    QuantType.MXFP6,
                    QuantType.MXFP8,
                ):
                    quant_weight, weight_scale, weight_global_scale = quantized
                else:
                    quant_weight, weight_scale = quantized
                self.fp8_scales_map[f"{name}.weight_scale"] = weight_scale
                bias = copy.deepcopy(module.bias) if module.bias is not None else None
                if self.quant_type in (
                    QuantType.NVFP4,
                    QuantType.MXFP4,
                    QuantType.MXFP6,
                    QuantType.MXFP8,
                ):
                    self.fp8_scales_map[f"{name}.weight_global_scale"] = weight_global_scale
                    quant_linear = self.quantize_linear_module(
                        weight=quant_weight,
                        weight_scale=weight_scale,
                        weight_global_scale=weight_global_scale,
                        bias=bias,
                        native_fp8_support=self.native_fp8_support,
                        quant_type=self.quant_type,
                    )
                else:
                    quant_linear = self.quantize_linear_module(
                        weight=quant_weight,
                        weight_scale=weight_scale,
                        bias=bias,
                        native_fp8_support=self.native_fp8_support,
                        quant_type=self.quant_type,
                    )
                replace_module(model, name, quant_linear)
                del module.weight, module.bias, module
                converted_count += 1
        print(f"[INIT] quantized linear layers: {converted_count}")
        cleanup_memory()

    def convert_linear(
        self, model: torch.nn.Module, scale: Optional[Union[torch.Tensor, float]] = None
    ):
        if scale is not None:
            self._convert_linear_with_scale(model, scale)
        else:
            self._convert_linear(model)

    def export_quantized_weight(self, model: torch.nn.Module, save_path: str):
        assert (
            self.quant_type == QuantType.FP8_PER_TENSOR
        ), "Currently only FP8_PER_TENSOR is supported for export"
        self.convert_linear(model)
        save_quantized_model(model, save_path, self.fp8_scales_map)
        logger.info(f"Quantized model saved to {save_path}")
        logger.info(f"Quantized scales saved to {save_path}/fp8_scales.safetensors")

    @staticmethod
    def load_quantized_model(model_class, save_path: str, device: str = "cpu"):
        """
        Load a quantized model from save_path.

        Args:
            model_class: The model class to instantiate
            save_path: Path to the saved model directory
            device: Device to load the model on

        Returns:
            Loaded model with quantized weights
        """
        return load_quantized_model(model_class, save_path, device)
