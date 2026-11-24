"""Encode initial local captions on GPU with one of two T5 weight groups loaded."""
import logging

import torch



def _weight_bytes(module):
    return sum(value.numel() * value.element_size()
               for value in [*module.parameters(), *module.buffers()])


def _weight_groups(model):
    if (getattr(model, "shared_pos", True) or getattr(model, "pos_embedding", None) is not None
            or len(getattr(model, "blocks", ())) < 2):
        raise ValueError("Grouped encoding requires the deployed UMT5 encoder with per-layer position embeddings")
    blocks = list(model.blocks)
    sizes = [_weight_bytes(block) for block in blocks]
    embedding, norm = _weight_bytes(model.token_embedding), _weight_bytes(model.norm)
    cut = min(range(1, len(blocks)),
              key=lambda n: max(embedding + sum(sizes[:n]), sum(sizes[n:]) + norm))
    return [([model.token_embedding, *blocks[:cut]], range(cut)),
            ([*blocks[cut:], model.norm], range(cut, len(blocks)))]


def _cpu_backing(model):
    return [(module, group, name, tensor.detach())
            for module in model.modules() for group in ("_parameters", "_buffers")
            for name, tensor in getattr(module, group, {}).items() if tensor is not None]


def _restore(backing, modules=None):
    selected = None if modules is None else {id(m) for top in modules for m in top.modules()}
    for module, group, name, tensor in backing:
        if selected is not None and id(module) not in selected:
            continue
        if group == "_parameters":
            module._parameters[name].data = tensor
        else:
            module._buffers[name] = tensor


def _encode_rows(model, ids, mask, states, layers, *, first, last, device):
    """Same batch-one forward operations and full sequence as the original T5."""
    values = []
    lengths = mask.gt(0).sum(dim=1).long().tolist()
    for index in range(len(ids)):
        row_mask = mask[index:index + 1].to(device)
        x = (model.dropout(model.token_embedding(ids[index:index + 1].to(device)))
             if first else states[index].to(device))
        for layer in layers:
            x = model.blocks[layer](x, row_mask, pos_bias=None)
        if last:
            x = model.dropout(model.norm(x))
            x[0, lengths[index]:] = 0.0
        values.append(x.detach().to("cpu"))
    return values


@torch.no_grad()
def encode_initial_locals(encoder, texts, device, *, max_length=128):
    """Caller holds the shared encoder lock. Never fall back to CPU neural work.

    Keep immutable CPU backing; each weight is uploaded once per preparation.
    No next-group prefetch, runtime empty_cache, video lock or video KV offload.
    """
    if not texts:
        return []
    if torch.device(device).type != "cuda":
        raise ValueError("Initial grouped local encoding requires a CUDA device")
    model = encoder.text_encoder
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("Initial grouped text encoding requires an evaluation-mode encoder")
    if any(p.device.type != "cpu" for p in encoder.parameters()):
        raise RuntimeError("Shared T5 must be restored to CPU before grouped preparation")
    if any("forge_backup_original_class" in m.__dict__ for m in encoder.modules()):
        raise RuntimeError("Remove dynamic parameter swapping before grouped preparation")
    groups = _weight_groups(model)
    backing = _cpu_backing(model)
    least_priority, _ = torch.cuda.Stream.priority_range()
    stream = torch.cuda.Stream(device=device, priority=least_priority)
    states = None
    stage_index = None
    try:
        ids, mask = encoder.tokenizer(texts, return_mask=True, add_special_tokens=True,
                                      padding="max_length", truncation=True, max_length=max_length)
        if ids.shape != mask.shape or tuple(ids.shape) != (len(texts), max_length):
            raise ValueError("Unexpected grouped tokenizer layout")
        for stage_index, (modules, layers) in enumerate(groups):
            with torch.cuda.stream(stream):
                for module in modules:
                    module.to(device=device)
                stream.synchronize()
                states = _encode_rows(model, ids, mask, states, layers,
                                      first=stage_index == 0, last=stage_index == len(groups) - 1,
                                      device=device)
                stream.synchronize()
            _restore(backing, modules)
        if any(row.ndim != 3 or row.shape[:2] != (1, max_length)
               or not torch.isfinite(row).all() for row in states):
            raise ValueError("Grouped local encoder returned invalid features")
        return states
    except Exception:
        logging.getLogger(__name__).exception("Grouped T5 encoding failed")
        raise
    finally:
        # Drain our own work before dropping GPU weights, including partial uploads.
        try:
            stream.synchronize()
        finally:
            _restore(backing)
