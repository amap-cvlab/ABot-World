from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def grouped_query_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Grouped-query attention using PyTorch SDPA.

    Tensors use the model-native [batch, heads, sequence, head_dim] layout.
    """
    repeats = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)
