"""Temporary GPU residency for initial encoding, before video generation starts."""
from contextlib import contextmanager, ExitStack
import logging


GIB = 1024 ** 3
ENCODING_HEADROOM_BYTES = 2 * GIB


@contextmanager
def initial_encoder_residency(encoder, device):
    """Borrow GPU memory for one initialization, then restore dynamic swapping.

    The caller holds the pipeline's encoder lock. Existing GPU-resident encoders
    and CPU-only callers keep their original placement. No second T5 is created.
    """
    modules = getattr(encoder, "modules", lambda: ())
    if str(device).split(":")[0] != "cuda":
        yield
        return

    import torch
    from abot_world.iris.utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb

    if not callable(getattr(encoder, "parameters", None)):
        yield
        return
    had_dynamic = any("forge_backup_original_class" in module.__dict__ for module in modules())
    tensors = [*encoder.parameters(), *encoder.buffers()]
    if tensors and all(t.device.type == "cuda" for t in tensors):
        yield
        return
    weight_bytes = sum(t.numel() * t.element_size() for t in tensors)
    del tensors  # Do not retain parameters across device moves.
    backing = [(module, group, name, tensor.detach())
               for module in modules() for group in ("_parameters", "_buffers")
               for name, tensor in getattr(module, group, {}).items()
               if tensor is not None]
    available_bytes = int(get_cuda_free_memory_gb(device) * GIB)
    required_bytes = weight_bytes + ENCODING_HEADROOM_BYTES
    fits = available_bytes >= required_bytes
    if not fits:
        if not had_dynamic:
            DynamicSwapInstaller.install_model(encoder, device=device)
        try:
            yield
        finally:
            if not had_dynamic:
                DynamicSwapInstaller.uninstall_model(encoder)
        return

    def restore():
        # Blocking CPU copies finish before video can use the freed memory.
        try:
            if backing:
                # Immutable CPU weights already exist: discard GPU copies.
                for module, group, name, tensor in backing:
                    if group == "_parameters":
                        module._parameters[name].data = tensor
                    else:
                        module._buffers[name] = tensor
            else:
                encoder.to(device="cpu")
        finally:
            if had_dynamic:
                DynamicSwapInstaller.install_model(encoder, device=device)
        torch.cuda.empty_cache()

    # ExitStack also restores after an encoding exception or a partial GPU move.
    # close() clears callbacks, so a failed restore is never attempted twice.
    with ExitStack() as cleanup:
        DynamicSwapInstaller.uninstall_model(encoder)
        cleanup.callback(restore)
        try:
            encoder.to(device=device)
        except torch.OutOfMemoryError:
            cleanup.close()
            if not had_dynamic:
                DynamicSwapInstaller.install_model(encoder, device=device)
                cleanup.callback(DynamicSwapInstaller.uninstall_model, encoder)
            logging.getLogger(__name__).warning("T5 GPU staging ran out of memory; using dynamic placement")
        yield
