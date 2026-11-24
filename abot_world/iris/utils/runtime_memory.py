"""Keep live text encoding on CPU; video caches are never moved or locked."""
from contextlib import contextmanager
import threading

import torch

_BATCH = threading.local()
CPU_ENCODER_THREADS = 4


def prepare_cpu_encoder(encoder):
    """Caller owns the shared encoder lock, outside video computation."""
    modules = list(getattr(encoder, "modules", lambda: ())())
    if not modules:
        return
    if any("forge_backup_original_class" in m.__dict__ for m in modules):
        from abot_world.iris.utils.memory import DynamicSwapInstaller
        DynamicSwapInstaller.uninstall_model(encoder)
    tensors = [*encoder.parameters(), *encoder.buffers()]
    if any(t.device.type != "cpu" for t in tensors):
        encoder.to(device="cpu")


@contextmanager
def cpu_encoder_batch(encoder):
    """Bound the CPU caller's intra-op team; never acquire a video lock.

    The caller holds its shared encoder lock. Nested local calls reuse the batch.
    CPU thread settings are restored on success, failure and cancellation.
    """
    if getattr(_BATCH, "active", False):
        yield
        return
    prepare_cpu_encoder(encoder)
    previous = torch.get_num_threads()
    changed = bool(getattr(encoder, "parameters", None)) and previous > CPU_ENCODER_THREADS
    if changed:
        torch.set_num_threads(CPU_ENCODER_THREADS)
    _BATCH.active = True
    try:
        yield
    finally:
        _BATCH.active = False
        if changed:
            torch.set_num_threads(previous)


def validate_quantization_device(config, device):
    """Reject unsupported per-token FP8 before allocating model weights."""
    if (config.get("use_fp8_gemm", True)
            and config.get("quant_type", "fp8-per-token") == "fp8-per-token"
            and torch.cuda.get_device_capability(device) < (8, 9)):
        raise ValueError(
            "fp8-per-token requires CUDA compute capability 8.9 or newer. "
            "Set quant_type: fp8-per-tensor in your YAML for the fallback, "
            "or use_fp8_gemm: false for BF16 (more GPU memory required)."
        )
