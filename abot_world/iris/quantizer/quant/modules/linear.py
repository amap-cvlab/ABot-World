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

import os
import time

import torch
from lightx2v_kernel.gemm import (
    cutlass_scaled_mxfp4_mm,
    cutlass_scaled_mxfp6_mxfp8_mm,
    cutlass_scaled_mxfp8_mm,
    cutlass_scaled_nvfp4_mm,
)

try:
    from torchao.quantization.utils import quant_int8_per_token_matmul as torchao_int8_gemm
    from torchao.quantization.utils import quantize_activation_per_token_absmax as torchao_int8_quant
except ImportError:
    try:
        from torchao.quantization.utils import _quant_int8_per_token_matmul as torchao_int8_gemm
        from torchao.quantization.utils import _quantize_activation_per_token_absmax as torchao_int8_quant
    except ImportError:
        torchao_int8_gemm, torchao_int8_quant = None, None

try:
    from vllm import _custom_ops as vllm_ops
except ImportError:
    vllm_ops = None

try:
    from ...kernels.python.sgl.int8_kernel import per_token_quant_int8 as sglang_int8_act_quant
except ImportError:
    sglang_int8_act_quant = None

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None

try:
    from q8_kernels.functional.linear import q8_linear
except ImportError:
    q8_linear = None

try:
    from ...kernels.python.mm.triton_kernels import (
        int8_gemm_bias_triton,
        int8_gemm_triton,
        int8_quantize_triton,
    )
except ImportError:
    int8_gemm_bias_triton, int8_gemm_triton, int8_quantize_triton = None, None, None

from ..quant_func import (
    fp8_gemm,
    fp8_per_block_quant,
    fp8_per_tensor_quant,
    fp8_per_token_group_quant,
    fp8_per_token_quant_sgl,
    fp8_weight_only_gemm,
    mxfp4_per_tensor_quant,
    mxfp6_per_tensor_quant,
    mxfp8_per_tensor_quant,
    nvfp4_per_tensor_quant,
)


# modified from https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,
        quant_type: str = "fp8-per-tensor",
        block_size: int = 128,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support
        self.quant_type = quant_type
        self.block_size = block_size
        self.profile_enabled = os.environ.get("ANGELSLIM_FP8_PROFILE", "0") == "1"

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        if self.profile_enabled and x.is_cuda:
            torch.cuda.synchronize(x.device)
        t0 = time.perf_counter()

        if self.quant_type == "fp8-per-tensor":
            origin_shape = None
            qinput, x_scale = fp8_per_tensor_quant(x)
        elif self.quant_type == "fp8-per-token":
            origin_shape = None
            x_2d = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_group_quant(x_2d, x_2d.shape[-1])
        elif self.quant_type == "fp8-per-token-sgl" and self.native_fp8_support:
            origin_shape = x.shape
            x_2d = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_quant_sgl(x_2d)
        elif self.quant_type == "fp8-per-block" and self.native_fp8_support:
            origin_shape = x.shape
            x = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_token_group_quant(
                x, group_size=128, column_major_scales=True, scale_tma_aligned=True
            )
        elif self.quant_type == "fp8-per-block" and not self.native_fp8_support:
            origin_shape = x.shape
            x_2d = x.view(-1, x.shape[-1])
            qinput, x_scale = fp8_per_block_quant(x_2d, block_size=128)
        elif self.quant_type == "fp8-per-channel-vllm":
            if vllm_ops is None:
                raise ImportError(
                    "quant_type='fp8-per-channel-vllm' requires vllm._custom_ops, but vllm is not installed"
                )
            origin_shape = x.shape if x.dim() == 3 else None
            x_2d = x.view(-1, x.shape[-1]) if x.dim() == 3 else x
            qinput, x_scale = vllm_ops.scaled_fp8_quant(
                x_2d, None, scale_ub=None, use_per_token_if_dynamic=True
            )
        else:
            raise ValueError(f"Invalid quant_type: {self.quant_type}")

        if self.profile_enabled and qinput.is_cuda:
            torch.cuda.synchronize(qinput.device)
        t1 = time.perf_counter()

        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            native_fp8_support=self.native_fp8_support,
            quant_type=self.quant_type,
            origin_shape=origin_shape,
        )

        if self.profile_enabled and output.is_cuda:
            torch.cuda.synchronize(output.device)
        t2 = time.perf_counter()

        if self.profile_enabled:
            qshape = tuple(qinput.shape)
            print(
                f"[FP8Linear:{self.quant_type}] quant_ms={(t1 - t0) * 1000:.3f}, "
                f"gemm_ms={(t2 - t1) * 1000:.3f}, qshape={qshape}"
            )

        if (
            self.quant_type in ["fp8-per-token", "fp8-per-token-sgl"]
            and x.dim() == 3
            and output.dim() == 2
        ):
            output = output.unsqueeze(0)

        # Restore original shape for fp8-per-block with native_fp8_support=False
        # (native_fp8_support=True case is handled in fp8_gemm_deepgemm_block)
        if (
            (
                (self.quant_type == "fp8-per-block" and not self.native_fp8_support)
                or self.quant_type == "fp8-per-channel-vllm"
            )
            and origin_shape is not None
            and len(origin_shape) == 3
            and output.dim() == 2
        ):
            output = output.view(origin_shape[0], origin_shape[1], -1)

        return output


class FP8WeightOnlyLinear(torch.nn.Module):
    """
    FP8 Weight-Only Quantized Linear Layer.

    This layer quantizes only the weights to FP8 while keeping activations
    in higher precision (bfloat16/float16). This provides a good balance
    between memory savings and accuracy.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,  # not used
        quant_type: str = "fp8-per-tensor-weight-only",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support  # not used
        self.quant_type = quant_type

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # For weight-only quantization, we don't quantize activations
        # Just use the original activations with quantized weights
        output = fp8_weight_only_gemm(
            A=x,  # Keep activations in original precision
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        return output


class INT8DynamicLinear(torch.nn.Module):
    """
    INT8 weight-only linear layer with per-channel scales.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        native_fp8_support: bool = False,  # not used
        quant_type: str = "int8",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support  # not used
        self.quant_type = quant_type

    @staticmethod
    def _is_backend_available(backend: str) -> bool:
        if backend == "torchao":
            return torchao_int8_quant is not None and torchao_int8_gemm is not None
        if backend == "vllm":
            return vllm_ops is not None and hasattr(torch.ops, "_C")
        if backend == "triton":
            return (
                int8_quantize_triton is not None
                and int8_gemm_triton is not None
                and int8_gemm_bias_triton is not None
            )
        if backend == "sgl":
            has_act = (
                sglang_int8_act_quant is not None
                or vllm_ops is not None
                or torchao_int8_quant is not None
                or int8_quantize_triton is not None
            )
            return sgl_kernel is not None and has_act
        if backend == "q8f":
            has_act = (
                vllm_ops is not None
                or torchao_int8_quant is not None
                or int8_quantize_triton is not None
            )
            return q8_linear is not None and has_act
        return False

    def _resolve_int8_backend(self) -> str:
        explicit_backend = {
            "int8-torchao": "torchao",
            "int8-vllm": "vllm",
            "int8-triton": "triton",
            "int8-sgl": "sgl",
            "int8-q8f": "q8f",
        }
        if self.quant_type in explicit_backend:
            backend = explicit_backend[self.quant_type]
            if not self._is_backend_available(backend):
                raise ImportError(
                    f"quant_type='{self.quant_type}' requires '{backend}' backend dependencies"
                )
            return backend

        # quant_type='int8' uses auto priority: sgl > vllm > torchao > triton
        for backend in ("sgl", "vllm", "torchao", "triton"):
            if self._is_backend_available(backend):
                return backend
        raise ImportError(
            "quant_type='int8' requires one of backends [sgl, vllm, torchao, triton], but none is available"
        )

    def _act_quant_int8_torchao(self, x_2d: torch.Tensor):
        input_tensor_quant, input_tensor_scale = torchao_int8_quant(x_2d)
        return input_tensor_quant, input_tensor_scale.float()

    def _act_quant_int8_vllm(self, x_2d: torch.Tensor):
        input_tensor_quant, input_tensor_scale, _ = vllm_ops.scaled_int8_quant(
            x_2d, scale=None, azp=None, symmetric=True
        )
        return input_tensor_quant, input_tensor_scale.float()

    def _act_quant_int8_triton(self, x_2d: torch.Tensor):
        input_tensor_quant, input_tensor_scale = int8_quantize_triton(x_2d)
        return input_tensor_quant, input_tensor_scale.float()

    def _act_quant_int8_sgl(self, x_2d: torch.Tensor):
        if sglang_int8_act_quant is not None:
            input_tensor_quant, input_tensor_scale = sglang_int8_act_quant(x_2d)
            return input_tensor_quant, input_tensor_scale.float()
        if vllm_ops is not None:
            return self._act_quant_int8_vllm(x_2d)
        if torchao_int8_quant is not None:
            return self._act_quant_int8_torchao(x_2d)
        if int8_quantize_triton is not None:
            return self._act_quant_int8_triton(x_2d)
        raise ImportError("int8-sgl activation quantization requires sglang/vllm/torchao/triton")

    def _act_quant_by_backend(self, x_2d: torch.Tensor, backend: str):
        if backend == "torchao":
            return self._act_quant_int8_torchao(x_2d)
        if backend == "vllm":
            return self._act_quant_int8_vllm(x_2d)
        if backend == "triton":
            return self._act_quant_int8_triton(x_2d)
        if backend == "sgl":
            return self._act_quant_int8_sgl(x_2d)
        if backend == "q8f":
            if vllm_ops is not None:
                return self._act_quant_int8_vllm(x_2d)
            if torchao_int8_quant is not None:
                return self._act_quant_int8_torchao(x_2d)
            return self._act_quant_int8_triton(x_2d)
        raise ValueError(f"Unsupported int8 backend: {backend}")

    def _gemm_int8_torchao(self, qinput, x_scale, out_dtype):
        output = torchao_int8_gemm(
            qinput,
            x_scale,
            self.weight.t(),
            self.weight_scale.t().float(),
            output_dtype=out_dtype,
        )
        if self.bias is not None:
            output.add_(self.bias.to(output.dtype))
        return output

    def _gemm_int8_vllm(self, qinput, x_scale, out_dtype):
        shape = (qinput.shape[0], self.weight.shape[0])
        output = torch.empty(shape, dtype=out_dtype, device=qinput.device, requires_grad=False)
        torch.ops._C.cutlass_scaled_mm(
            output,
            qinput,
            self.weight.t(),
            x_scale,
            self.weight_scale.t(),
            self.bias,
        )
        return output

    def _gemm_int8_triton(self, qinput, x_scale, out_dtype):
        if self.bias is not None:
            return int8_gemm_bias_triton(
                qinput,
                self.weight,
                self.bias,
                x_scale,
                self.weight_scale,
                output_dtype=out_dtype,
            )
        return int8_gemm_triton(
            qinput,
            self.weight,
            x_scale,
            self.weight_scale,
            output_dtype=out_dtype,
        )

    def _gemm_int8_sgl(self, qinput, x_scale, out_dtype):
        return sgl_kernel.int8_scaled_mm(
            qinput,
            self.weight.t(),
            x_scale,
            self.weight_scale.t(),
            out_dtype,
            self.bias,
        )

    def _gemm_int8_q8f(self, qinput, x_scale, out_dtype):
        bias_fp32 = self.bias.float() if self.bias is not None else None
        return q8_linear(
            qinput,
            self.weight,
            bias_fp32,
            x_scale.float(),
            self.weight_scale,
            fuse_gelu=False,
            out_dtype=out_dtype,
        )

    def _gemm_by_backend(self, qinput, x_scale, out_dtype, backend: str):
        if backend == "torchao":
            return self._gemm_int8_torchao(qinput, x_scale, out_dtype)
        if backend == "vllm":
            return self._gemm_int8_vllm(qinput, x_scale, out_dtype)
        if backend == "triton":
            return self._gemm_int8_triton(qinput, x_scale, out_dtype)
        if backend == "sgl":
            return self._gemm_int8_sgl(qinput, x_scale, out_dtype)
        if backend == "q8f":
            return self._gemm_int8_q8f(qinput, x_scale, out_dtype)
        raise ValueError(f"Unsupported int8 backend: {backend}")

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        need_reshape = x.dim() == 3
        if need_reshape:
            origin_shape = x.shape
            x_2d = x.view(-1, x.shape[-1])
        else:
            origin_shape = None
            x_2d = x

        backend = self._resolve_int8_backend()
        qinput, x_scale = self._act_quant_by_backend(x_2d, backend)
        output = self._gemm_by_backend(qinput, x_scale, x.dtype, backend)

        if need_reshape and output.dim() == 2:
            output = output.view(origin_shape[0], origin_shape[1], -1)
        return output.to(ori_dtype)


class FP4DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        weight_global_scale: torch.Tensor = None,
        native_fp8_support: bool = False,
        quant_type: str = "nvfp4",
        block_size: int = 16,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.native_fp8_support = native_fp8_support
        self.quant_type = quant_type
        self.block_size = block_size
        self.profile_enabled = os.environ.get("ANGELSLIM_NVFP4_PROFILE", "0") == "1"
        if weight_global_scale is None:
            weight_global_scale = torch.tensor(1.0, dtype=torch.float32, device=weight.device)
        self.weight_global_scale = torch.nn.Parameter(
            weight_global_scale.to(dtype=torch.float32), requires_grad=False
        )
        self.calibrate_x_absmax()

    def calibrate_x_absmax(self):
        if self.quant_type in ("mxfp4", "mxfp6", "mxfp8"):
            self.x_absmax = torch.tensor(1.0, dtype=torch.float32, device=self.weight.device)
            self.input_global_scale = torch.tensor(
                1.0, dtype=torch.float32, device=self.weight.device
            )
        else:
            self.x_absmax = torch.tensor(5.0, dtype=torch.float32, device=self.weight.device)
            self.input_global_scale = (2688.0 / self.x_absmax).to(torch.float32)
        self.alpha = 1.0 / (self.input_global_scale * self.weight_global_scale)

    @torch.compiler.disable(recursive=True)
    def forward(self, x):
        ori_dtype = x.dtype
        assert ori_dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "x.dtype must be float32, bfloat16, or float16"

        if ori_dtype == torch.float32:
            x = x.to(torch.bfloat16)

        need_reshape = x.dim() == 3
        if need_reshape:
            origin_shape = x.shape
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x

        if self.profile_enabled and x_2d.is_cuda:
            torch.cuda.synchronize(x_2d.device)
        t0 = time.perf_counter()
        if self.quant_type == "nvfp4":
            qinput, x_scale, _ = nvfp4_per_tensor_quant(x_2d, self.input_global_scale)
            output = cutlass_scaled_nvfp4_mm(
                qinput,
                self.weight,
                x_scale,
                self.weight_scale,
                self.alpha,
                bias=self.bias,
            )
        elif self.quant_type == "mxfp4":
            qinput, x_scale, _ = mxfp4_per_tensor_quant(x_2d)
            output = cutlass_scaled_mxfp4_mm(
                qinput,
                self.weight,
                x_scale,
                self.weight_scale,
                self.alpha,
                bias=self.bias,
            )
        elif self.quant_type == "mxfp8":
            qinput, x_scale, _ = mxfp8_per_tensor_quant(x_2d)
            output = cutlass_scaled_mxfp8_mm(
                qinput,
                self.weight,
                x_scale,
                self.weight_scale,
                self.alpha,
                bias=self.bias,
            )
        elif self.quant_type == "mxfp6":
            qinput, x_scale, _ = mxfp8_per_tensor_quant(x_2d)
            output = cutlass_scaled_mxfp6_mxfp8_mm(
                qinput,
                self.weight,
                x_scale,
                self.weight_scale,
                self.alpha,
                bias=self.bias,
            )
        else:
            raise ValueError(f"Invalid quant_type for FP4DynamicLinear: {self.quant_type}")
        if self.profile_enabled and x_2d.is_cuda:
            torch.cuda.synchronize(x_2d.device)
        t1 = time.perf_counter()
        if self.profile_enabled and x_2d.is_cuda:
            torch.cuda.synchronize(x_2d.device)
        t2 = time.perf_counter()

        if self.profile_enabled:
            print(
                f"[NVFP4Linear] quant_ms={(t1 - t0) * 1000:.3f}, "
                f"gemm_ms={(t2 - t1) * 1000:.3f}, "
                f"shape=({x_2d.shape[0]}, {x_2d.shape[1]})"
            )

        if need_reshape:
            output = output.view(origin_shape[0], origin_shape[1], -1)

        return output.to(ori_dtype)
