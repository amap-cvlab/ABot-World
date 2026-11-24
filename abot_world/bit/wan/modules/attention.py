# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import warnings

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import aiter
    AITER_AVAILABLE = True
except ModuleNotFoundError:
    AITER_AVAILABLE = False


def initialize_attention_implementation():
    requested = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION")
    if requested is not None:
        return requested.lower()
    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_implementation()

if ATTENTION_IMPLEMENTATION == "aiter":
    if not AITER_AVAILABLE:
        raise ImportError(
            "DIFFSYNTH_ATTENTION_IMPLEMENTATION=aiter, but the aiter package "
            "is unavailable"
        )
    if not hasattr(aiter, "flash_attn_func"):
        raise ImportError("aiter.flash_attn_func is unavailable")

__all__ = [
    'flash_attention',
    'attention',
    'aiter_attention',
    'ATTENTION_IMPLEMENTATION',
]


def _has_padded_lengths(lengths, full_length):
    if lengths is None:
        return False
    lengths = torch.as_tensor(lengths, device="cpu").reshape(-1)
    return bool((lengths != int(full_length)).any().item())


def aiter_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """Run AITER dense attention on Wan-layout [B, L, H, D] tensors."""

    if not AITER_AVAILABLE or not hasattr(aiter, "flash_attn_func"):
        raise ImportError("aiter.flash_attn_func is unavailable")
    if _has_padded_lengths(q_lens, q.shape[1]) or _has_padded_lengths(
        k_lens, k.shape[1]
    ):
        raise ValueError(
            "AITER attention does not support padded q_lens/k_lens in this path"
        )
    if dropout_p != 0:
        raise ValueError("AITER attention currently requires dropout_p=0")

    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError(f"AITER attention dtype must be fp16 or bf16, got {dtype}")
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q, k, v = half(q), half(k), half(v)
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    if len(window_size) == 2:
        aiter_window_size = (window_size[0], window_size[1], 0)
    elif len(window_size) == 3:
        aiter_window_size = tuple(window_size)
    else:
        raise ValueError(
            f"AITER window_size must contain two or three values, got {window_size}"
        )

    requires_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    )
    out = aiter.flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=aiter_window_size,
        deterministic=deterministic,
        # AITER backward reads LSE from its autograd context.
        return_lse=requires_grad,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if ATTENTION_IMPLEMENTATION == "aiter":
        return aiter_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
        )
    if not FLASH_ATTN_3_AVAILABLE and not FLASH_ATTN_2_AVAILABLE:
        warnings.warn(
            'Flash attention is not available, use scaled_dot_product_attention instead.'
        )
        return attention(q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic, dtype=dtype)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if (
        ATTENTION_IMPLEMENTATION == "aiter"
        or FLASH_ATTN_2_AVAILABLE
        or FLASH_ATTN_3_AVAILABLE
    ):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
