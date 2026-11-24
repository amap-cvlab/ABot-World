from contextlib import nullcontext

import torch
import triton


def get_device_type():
    if torch.cuda.is_available():
        try:
            if torch.version.hip is not None:
                return "hip"
        except AttributeError:
            pass
        return "cuda"

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except (AttributeError, RuntimeError):
        pass

    return "cpu"


def get_device_count(device_type):
    if device_type == "cuda" or device_type == "hip":
        return torch.cuda.device_count()
    elif device_type == "xpu":
        try:
            return torch.xpu.device_count()
        except (AttributeError, RuntimeError):
            return 0
    return 0


MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2
DEVICE_TYPE = get_device_type()
DEVICE_COUNT = get_device_count(DEVICE_TYPE)

if DEVICE_COUNT > 1:
    if DEVICE_TYPE in ("cuda", "hip"):
        torch_gpu_device = torch.cuda.device
    elif DEVICE_TYPE == "xpu":
        torch_gpu_device = torch.xpu.device
else:

    def torch_gpu_device(device):
        return nullcontext()


def calculate_settings(
    n: int,
) -> (
    int,
    int,
):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps
