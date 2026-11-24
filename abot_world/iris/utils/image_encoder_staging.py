"""Temporarily stage Wan's image-encoding path without moving its decoder."""
from contextlib import contextmanager, ExitStack

import torch



@contextmanager
def image_encoder_residency(vae, device):
    """Use GPU for encoding, restoring the original placement even on failure.

    Only Wan's encoder and quantization convolution participate in encoding.
    CPU callers and wrappers without that separate path retain their placement.
    The caller transfers the resulting latents to the inference device.
    """
    target = torch.device(device)
    model = getattr(vae, "model", None)
    parts = [getattr(model, name, None) for name in ("encoder", "conv1")]
    if target.type != "cuda" or not all(isinstance(part, torch.nn.Module) for part in parts):
        yield next(vae.parameters()).device
        return

    moves = [(part, next(part.parameters()).device) for part in parts]
    moves = [(part, original) for part, original in moves if original != target]
    if not moves:
        yield target
        return

    # Register every restoration before any move: a partial GPU transfer can fail.
    # ExitStack attempts all restorations if one of them raises.
    with ExitStack() as cleanup:
        cleanup.callback(torch.cuda.empty_cache)
        for part, original in moves:
            cleanup.callback(part.to, device=original)
        for part, _ in moves:
            part.to(device=target)
        yield target
