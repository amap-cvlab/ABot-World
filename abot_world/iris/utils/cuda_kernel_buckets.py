"""
将 PyTorch CUDA profiler 事件按内核/算子名称粗分为若干类，便于估算
memcpy、GEMM/Linear、Attention、Norm 等在 GPU 时间中的占比。

说明：
- 分类基于名称子串启发式，不同 CUDA / Triton / cuBLAS 版本下内核名会有差异；
- 若需更细粒度，请配合 prof.export_chrome_trace() 在 chrome://tracing 或 Perfetto 中查看。
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

# 顺序有意义：先匹配更具体的类别
_BUCKET_ORDER: List[str] = [
    "memcpy_sync",
    "attention",
    "gemm_linear",
    "conv",
    "norm",
    "elementwise_reduce",
    "other",
]


def classify_cuda_name(name: str) -> str:
    """根据内核或 aten 算子名称归入粗分类。"""
    n = (name or "").lower()

    # 设备同步、拷贝、显存设置（含部分 launch 开销）
    if any(
        s in n
        for s in (
            "memcpy",
            "memset",
            "memgetinfo",
            "cudaget",
            "cudaevent",
            "cudaoccupancy",
            "cuda stream",
            "cudastream",
            "cudadevicesynchronize",
            "cuda devicesynchronize",
            "device synchronize",
            "eventrecord",
            "eventsynchronize",
        )
    ):
        return "memcpy_sync"

    # FlashAttention / SDPA / 各类 fused attention
    if any(
        s in n
        for s in (
            "flash_attn",
            "flash-attn",
            "scaled_dot_product",
            "sdpa",
            "efficient_attention",
            "mem_eff_attention",
            "fused_attention",
            "fmha",
            "mha_default",
            "multi_head_attention",
            "contrib_attn",
            "attention_forward",
            "attention_backward",
            "softmax",
        )
    ):
        # 独立 softmax 小核也可能与 attention 同桶；若需拆开可把 softmax 挪到 elementwise
        return "attention"

    # MatMul / Linear / GEMM（含 cutlass、cublas、triton、fp8）
    if any(
        s in n
        for s in (
            "gemm",
            "cublas",
            "cutlass",
            "matmul",
            "mat_mul",
            "aten::linear",
            "aten::mm",
            "aten::bmm",
            "aten::addmm",
            "aten::matmul",
            "scaled_mm",
            "fp8",
            "wmma",
            "mma_sync",
            "triton",
            "dot",
        )
    ):
        return "gemm_linear"

    if any(s in n for s in ("conv", "cudnn", "depthwise", "convolution")):
        return "conv"

    if any(
        s in n
        for s in (
            "layernorm",
            "layer_norm",
            "rms_norm",
            "group_norm",
            "flash_norm",
            "aten::layer_norm",
            "aten::group_norm",
            "aten::native_layer_norm",
        )
    ):
        return "norm"

    if any(
        s in n
        for s in (
            "elementwise",
            "vectorized",
            "unary",
            "binary",
            "reduce",
            "reduction",
            "activation",
            "silu",
            "gelu",
            "swiglu",
            "relu",
            "aten::add",
            "aten::mul",
            "aten::div",
            "aten::pow",
            "aten::sqrt",
            "aten::rsqrt",
        )
    ):
        return "elementwise_reduce"

    return "other"


def _event_cuda_time_us(event) -> float:
    for attr in (
        "cuda_time_total",
        "self_cuda_time_total",
        "self_cuda_time_total_us",
    ):
        v = getattr(event, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def aggregate_from_profiler(prof) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    """
    从 torch.profiler.profile 实例聚合：
    - 返回 (bucket -> 微秒总和, 按耗时排序的 (name, us) 列表)
    使用 prof.events() 以尽量接近 CUDA 内核级名称。
    """
    bucket_us: DefaultDict[str, float] = defaultdict(float)
    per_name: DefaultDict[str, float] = defaultdict(float)

    try:
        events = prof.events()
    except Exception:
        events = []

    for e in events:
        us = _event_cuda_time_us(e)
        if us <= 0:
            continue
        name = getattr(e, "name", "") or ""
        b = classify_cuda_name(name)
        bucket_us[b] += us
        per_name[name] += us

    # 若 events() 为空，回退到 key_averages（多为 aten 级）
    if not bucket_us and not per_name:
        try:
            for avg in prof.key_averages():
                us = float(getattr(avg, "cuda_time_total", 0) or getattr(avg, "self_cuda_time_total", 0) or 0)
                if us <= 0:
                    continue
                name = getattr(avg, "key", "") or ""
                b = classify_cuda_name(name)
                bucket_us[b] += us
                per_name[name] += us
        except Exception:
            pass

    sorted_names = sorted(per_name.items(), key=lambda x: -x[1])
    out = {k: bucket_us.get(k, 0.0) for k in _BUCKET_ORDER}
    for k, v in bucket_us.items():
        if k not in out:
            out[k] = v
    return out, sorted_names


def format_bucket_report(
    bucket_us: Dict[str, float],
    top_names: Iterable[Tuple[str, float]],
    top_k: int = 25,
) -> str:
    total = sum(bucket_us.values()) or 1e-9
    lines: List[str] = []
    lines.append("=== CUDA 时间按粗分类（微秒 / 占比）===")
    for k in _BUCKET_ORDER:
        if k in bucket_us and bucket_us[k] > 0:
            us = bucket_us[k]
            lines.append(f"  {k:22s}  {us:12.1f} us  ({100.0 * us / total:5.1f}%)")
    # 其它未在顺序表中的桶
    for k, us in sorted(bucket_us.items(), key=lambda x: -x[1]):
        if k in _BUCKET_ORDER:
            continue
        if us > 0:
            lines.append(f"  {k:22s}  {us:12.1f} us  ({100.0 * us / total:5.1f}%)")
    lines.append(f"  {'TOTAL':22s}  {total:12.1f} us")
    lines.append("")
    lines.append(f"=== 耗时最高的 {top_k} 个事件名（微秒）===")
    for i, (name, us) in enumerate(top_names):
        if i >= top_k:
            break
        short = name if len(name) <= 120 else name[:117] + "..."
        lines.append(f"  {us:10.1f}  {short}")
    return "\n".join(lines)
