# causal_wan2_2.py
# 基于 Wan 2.2 结构 + Wan 2.1 Causal 版本改造

import math
import os
from typing import Any
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules.model import SimpleAdapter, MLPProj, WanI2VCrossAttention

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

from wan.modules.attention import attention  # 你 2.1 causal 版本里用的那个 attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    rope_params,
    sinusoidal_embedding_1d,
    WanSelfAttention,
)



# ===== Debug helpers：只在异常/即将越界时打印，正常路径不刷日志 =====

def _dbg_tensor(name, x):
    if torch.is_tensor(x):
        return f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
    return f"{name}: {type(x)}={x}"


def _dbg_print(tag, **kwargs):
    print(f"\n[DEBUG][{tag}]")
    for k, v in kwargs.items():
        try:
            print(" ", _dbg_tensor(k, v))
        except Exception as e:
            print(f"  {k}: <print failed: {e}>")


def _dbg_block_mask(mask):
    # BlockMask 不同 PyTorch 版本内部字段不稳定，这里只安全打印类型和 repr。
    try:
        return repr(mask)
    except Exception as e:
        return f"<BlockMask repr failed: {e}>"


def _is_checkpoint_stop_signal(err: BaseException) -> bool:
    """gradient checkpoint 重算时使用的内部控制流，不是真实错误；勿打印 DEBUG failed。"""
    return type(err).__name__ == "_StopRecomputationError"


# ===== 新增：带 start_frame 的 causal_rope_apply =====

@torch.amp.autocast('cuda', enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    与 2.1 causal 版本一致：在时间维上加入起始帧偏移，用于推理时逐帧累积。
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        try:
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, -1)

            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        except Exception as e:
            if _is_checkpoint_stop_signal(e):
                raise
            _dbg_print(
                "causal_rope_apply.failed",
                error=repr(e),
                x=x,
                grid_sizes=grid_sizes,
                freqs=freqs,
                start_frame=start_frame,
                batch_index=i,
                f=f,
                h=h,
                w=w,
                seq_len=seq_len,
                n=n,
                c=c,
            )
            raise
    return torch.stack(output).float()


# Relative-RoPE freqs are keyed by (f, h, w, clamped frame ids, head dim,
# device). In steady state the visible window size and frame ids are constant,
# so the expanded [seq_len, 1, dim] complex tensor is identical every block and
# can be reused instead of rebuilt (arange/clamp/expand/cat/reshape).
_REL_FREQS_I_CACHE: dict = {}


def _get_relative_freqs_i(freqs, f, h, w, t_index, device):
    c = int(freqs.size(1))
    cache_key = (int(f), int(h), int(w), tuple(t_index.tolist()), c, str(device))
    cached = _REL_FREQS_I_CACHE.get(cache_key)
    if cached is not None:
        return cached

    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_i = torch.cat([
        freqs_split[0][t_index].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(torch.complex64)
    _REL_FREQS_I_CACHE[cache_key] = freqs_i
    return freqs_i


@torch.amp.autocast('cuda', enabled=False)
def relative_rope_apply(x, grid_sizes, freqs, frame_indices=None):
    """
    Apply RoPE with frame ids local to the currently visible attention window.

    KV-cache inference can evict old frames. When relative RoPE is enabled we
    keep raw keys in cache and re-apply RoPE to the visible window so the query
    and keys share a stable local coordinate system instead of growing absolute
    frame ids forever.
    """
    n = x.size(2)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        # fp32/complex64 RoPE: numerically sufficient, ~half the memory traffic
        # of the previous fp64/complex128 path.
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        )
        if frame_indices is None:
            t_index = torch.arange(f, device=x.device, dtype=torch.long)
        else:
            t_index = frame_indices[:f].to(device=x.device, dtype=torch.long)
        # Echo/Infinity-style block-relative RoPE keeps temporal ids inside the
        # visible local window. For local_attn_size=21, the max valid id is 20.
        t_index = torch.clamp(t_index, min=0, max=20)

        freqs_i = _get_relative_freqs_i(freqs, f, h, w, t_index, x.device)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def freqs_at(indices, dim, theta=10000, device='cpu'):
    assert dim % 2 == 0
    t = torch.tensor(indices, dtype=torch.float64, device=device)
    freqs = torch.outer(
        t,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=device, dtype=torch.float64).div(dim))
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply_with_refimg(x, freqs, num_heads):
    if x.dim() == 3:
        b, s, _ = x.shape
        x = x.view(b, s, num_heads, -1)
    # RoPE in fp32 (complex64) is numerically sufficient and roughly halves the
    # memory traffic vs. the previous fp64/complex128 path.
    x_out = torch.view_as_complex(x.to(torch.float32).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2
    ))
    freqs = freqs.to(device=x.device)
    if freqs.dtype != torch.complex64:
        freqs = freqs.to(torch.complex64)
    x_out = torch.view_as_real(x_out * freqs).flatten(3)
    return x_out.to(x.dtype)


# Ref-image RoPE freqs depend only on (num_slots, tokens_per_slot, ref grid,
# head dim, device); they are constant across denoising blocks, so cache them
# instead of rebuilding every call.
_REF_FREQS_CACHE: dict = {}


# Optimization E: in the relative-RoPE KV-cache path, cache the post-RoPE video
# keys and only re-rope the new block's query each step, instead of re-roping
# the whole visible window every block. This relies on RoPE attention logits
# depending only on query-key position *differences*: with a local window of
# <=local_attn_size frames the window-local ids never exceed the trained range
# (the clamp is a no-op), so an absolute counter with periodic re-basing yields
# identical logits. We re-base (re-rope the window once) before the counter
# nears the rotary table limit (1024). Only valid when sink_size == 0.
_REL_ROPE_CACHE_ENABLED = os.environ.get("REL_ROPE_CACHE", "1") != "0"
# Keep roped positions well below the 1024-entry rotary table. Comparable in
# magnitude to the absolute-RoPE path for short runs, so quality is unaffected.
_REL_ROPE_REBASE_MAX_POS = int(os.environ.get("REL_ROPE_REBASE_MAX_POS", "256"))
_REL_ROPE_DEBUG = os.environ.get("REL_ROPE_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
_REL_ROPE_DEBUG_LIMIT = int(os.environ.get("REL_ROPE_DEBUG_LIMIT", "50"))
_REL_ROPE_DEBUG_COUNT = 0
_REL_ROPE_DEBUG_ONCE_KEYS = set()


def _rel_rope_debug_print(tag, once_key=None, **kwargs):
    global _REL_ROPE_DEBUG_COUNT
    if not _REL_ROPE_DEBUG or _REL_ROPE_DEBUG_COUNT >= _REL_ROPE_DEBUG_LIMIT:
        return
    if once_key is not None:
        if once_key in _REL_ROPE_DEBUG_ONCE_KEYS:
            return
        _REL_ROPE_DEBUG_ONCE_KEYS.add(once_key)
    _REL_ROPE_DEBUG_COUNT += 1
    parts = []
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = value.item()
            else:
                value = tuple(value.shape)
        parts.append(f"{key}={value}")
    print(f"[REL_ROPE_DEBUG][{tag}] " + " ".join(parts), flush=True)


def _build_ref_freqs(freqs, num_slots, tokens_per_slot, ref_grid, device):
    patch_t, patch_h, patch_w = [int(v) for v in ref_grid]
    freq_dim = int(freqs.shape[1])
    cache_key = (
        int(num_slots), int(tokens_per_slot), patch_t, patch_h, patch_w,
        freq_dim, str(device),
    )
    cached = _REF_FREQS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    f_band = freq_dim - 2 * (freq_dim // 3)
    h_band = freq_dim // 3
    w_band = freq_dim // 3
    temporal_step = max(int(tokens_per_slot), 256)
    neg_temporal = [-(int(num_slots) - i) * temporal_step for i in range(int(num_slots))]
    t_freqs = freqs_at(neg_temporal, 2 * f_band, device=device)
    freqs_split = freqs.split([f_band, h_band, w_band], dim=1)
    h_freqs = freqs_split[1][:patch_h].to(device)
    w_freqs = freqs_split[2][:patch_w].to(device)
    ref_freqs = torch.cat([
        t_freqs[:, None, None, None, :].expand(num_slots, patch_t, patch_h, patch_w, f_band),
        h_freqs[None, None, :, None, :].expand(num_slots, patch_t, patch_h, patch_w, h_band),
        w_freqs[None, None, None, :, :].expand(num_slots, patch_t, patch_h, patch_w, w_band),
    ], dim=-1).reshape(int(num_slots) * int(tokens_per_slot), 1, -1).to(torch.complex64)
    _REF_FREQS_CACHE[cache_key] = ref_freqs
    return ref_freqs


# ===== Causal Self-Attention（替换原 WanSelfAttention） =====

# Keep the causal-forcing flex attention path on the default inductor mode.
# max-autotune can fail on BlockMask symbolic sparse shapes during inference.
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

# #Casual Forcing 配置
# flex_attention = torch.compile(
#     flex_attention,
#     dynamic=False,
#     mode="default"
# )


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 use_relative_rope=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_relative_rope = bool(use_relative_rope)
        # 注意：max_attention_size 只用于 KV cache 推理路径
        self.max_attention_size = 880 * 21 if local_attn_size == -1 else local_attn_size * 880
        # self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        print(
            f"CausalWanSelfAttention: max_attention_size={self.max_attention_size}, local_attn_size={self.local_attn_size}, sink_size={self.sink_size}")
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _run_flex_attention(
            self,
            *,
            query,
            key,
            value,
            block_mask,
            padded_length=0,
            tag="self_attn.flex_attention.failed",
            **debug_ctx,
    ):
        """
        flex_attention 的薄封装：正常时不打印；只在 kernel/shape/mask 崩时打印关键上下文。
        """
        try:
            out = flex_attention(
                query=query.transpose(2, 1),
                key=key.transpose(2, 1),
                value=value.transpose(2, 1),
                block_mask=block_mask,
            )
            if padded_length > 0:
                out = out[:, :, :-padded_length]
            return out.transpose(2, 1)
        except Exception as e:
            if _is_checkpoint_stop_signal(e):
                raise
            _dbg_print(
                tag,
                error=repr(e),
                query=query,
                key=key,
                value=value,
                query_t=query.transpose(2, 1),
                key_t=key.transpose(2, 1),
                value_t=value.transpose(2, 1),
                block_mask=_dbg_block_mask(block_mask),
                padded_length=padded_length,
                **debug_ctx,
            )
            raise

    def forward(
            self,
            x,  # [B, L, C]
            seq_lens,
            grid_sizes,
            freqs,
            block_mask: BlockMask | None = None,
            kv_cache: dict | None = None,
            current_start: int = 0,
            cache_start: int | None = None,
    ):
        """
        训练：kv_cache is None，使用 flex_attention + block_mask
        推理：kv_cache not None，使用显式 KV cache + attention()
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x_):
            q = self.norm_q(self.q(x_)).view(b, s, n, d)
            k = self.norm_k(self.k(x_)).view(b, s, n, d)
            v = self.v(x_).view(b, s, n, d)
            return q, k, v

        try:
            q, k, v = qkv_fn(x)
        except Exception as e:
            if _is_checkpoint_stop_signal(e):
                raise
            _dbg_print(
                "self_attn.qkv.failed",
                error=repr(e),
                x=x,
                b=b,
                s=s,
                n=n,
                d=d,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
            )
            raise

        if kv_cache is None:
            frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
            current_start_frame = current_start // frame_seqlen if frame_seqlen > 0 else 0
            num_ref = int(getattr(self, "_num_ref_tokens", 0) or 0)
            try:
                is_tf = bool(getattr(self, "_is_teacher_forcing", False)) or (
                    num_ref == 0 and s == seq_lens[0].item() * 2
                ) or (
                    num_ref > 0 and s == num_ref + seq_lens[0].item() * 2
                )
            except Exception:
                is_tf = False

            if num_ref > 0:
                ref_info = {
                    "num_slots": int(getattr(self, "_ref_num_slots", 0) or 0),
                    "tokens_per_slot": int(getattr(self, "_ref_tokens_per_frame", 0) or 0),
                    "grid": getattr(self, "_ref_grid_sizes", None),
                }
                if ref_info["num_slots"] <= 0 or ref_info["tokens_per_slot"] <= 0 or ref_info["grid"] is None:
                    raise RuntimeError("Ref tokens are present but ref RoPE metadata is incomplete.")
                ref_freqs = _build_ref_freqs(
                    freqs=freqs,
                    num_slots=ref_info["num_slots"],
                    tokens_per_slot=ref_info["tokens_per_slot"],
                    ref_grid=ref_info["grid"],
                    device=q.device,
                )

                if is_tf:
                    branch_len = (s - num_ref) // 2
                    roped_query = torch.cat([
                        rope_apply_with_refimg(q[:, :num_ref], ref_freqs, self.num_heads),
                        rope_apply(q[:, num_ref:num_ref + branch_len], grid_sizes, freqs),
                        rope_apply(q[:, num_ref + branch_len:], grid_sizes, freqs),
                    ], dim=1).type_as(v)
                    roped_key = torch.cat([
                        rope_apply_with_refimg(k[:, :num_ref], ref_freqs, self.num_heads),
                        rope_apply(k[:, num_ref:num_ref + branch_len], grid_sizes, freqs),
                        rope_apply(k[:, num_ref + branch_len:], grid_sizes, freqs),
                    ], dim=1).type_as(v)
                else:
                    roped_query = torch.cat([
                        rope_apply_with_refimg(q[:, :num_ref], ref_freqs, self.num_heads),
                        causal_rope_apply(
                            q[:, num_ref:],
                            grid_sizes,
                            freqs,
                            start_frame=current_start_frame,
                        ),
                    ], dim=1).type_as(v)
                    roped_key = torch.cat([
                        rope_apply_with_refimg(k[:, :num_ref], ref_freqs, self.num_heads),
                        causal_rope_apply(
                            k[:, num_ref:],
                            grid_sizes,
                            freqs,
                            start_frame=current_start_frame,
                        ),
                    ], dim=1).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                if padded_length > 0:
                    roped_query = torch.cat(
                        [roped_query, roped_query.new_zeros(q.shape[0], padded_length, q.shape[2], q.shape[3])],
                        dim=1,
                    )
                    roped_key = torch.cat(
                        [roped_key, roped_key.new_zeros(k.shape[0], padded_length, k.shape[2], k.shape[3])],
                        dim=1,
                    )
                    v = torch.cat(
                        [v, v.new_zeros(v.shape[0], padded_length, v.shape[2], v.shape[3])],
                        dim=1,
                    )
                x = self._run_flex_attention(
                    query=roped_query,
                    key=roped_key,
                    value=v,
                    block_mask=block_mask,
                    padded_length=padded_length,
                    x=x,
                    q=q,
                    k=k,
                    v=v,
                    roped_query=roped_query,
                    roped_key=roped_key,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    is_tf=is_tf,
                )
            elif is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    try:
                        rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                        rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    except Exception as e:
                        if _is_checkpoint_stop_signal(e):
                            raise
                        _dbg_print(
                            "self_attn.rope_apply.tf.failed",
                            error=repr(e),
                            ii=ii,
                            q=q,
                            k=k,
                            v=v,
                            q_chunk=q_chunk[ii],
                            k_chunk=k_chunk[ii],
                            grid_sizes=grid_sizes,
                            freqs=freqs,
                            seq_lens=seq_lens,
                        )
                        raise
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                if padded_length > 0:
                    padded_roped_query = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_roped_key = torch.cat(
                        [roped_key,
                         torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                     device=k.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )

                    x = self._run_flex_attention(
                        query=padded_roped_query,
                        key=padded_roped_key,
                        value=padded_v,
                        block_mask=block_mask,
                        padded_length=padded_length,
                        x=x,
                        q=q,
                        k=k,
                        v=v,
                        roped_query=roped_query,
                        roped_key=roped_key,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        is_tf=is_tf,
                    )
                else:
                    x = self._run_flex_attention(
                        query=roped_query,
                        key=roped_key,
                        value=v,
                        block_mask=block_mask,
                        padded_length=0,
                        x=x,
                        q=q,
                        k=k,
                        v=v,
                        roped_query=roped_query,
                        roped_key=roped_key,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        is_tf=is_tf,
                    )

            else:
                try:
                    roped_query = causal_rope_apply(
                        q,
                        grid_sizes,
                        freqs,
                        start_frame=current_start_frame,
                    ).type_as(v)
                    roped_key = causal_rope_apply(
                        k,
                        grid_sizes,
                        freqs,
                        start_frame=current_start_frame,
                    ).type_as(v)
                except Exception as e:
                    if _is_checkpoint_stop_signal(e):
                        raise
                    _dbg_print(
                        "self_attn.rope_apply.failed",
                        error=repr(e),
                        x=x,
                        q=q,
                        k=k,
                        v=v,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                        seq_lens=seq_lens,
                    )
                    raise

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                if padded_length > 0:
                    padded_roped_query = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_roped_key = torch.cat(
                        [roped_key,
                         torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                     device=k.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )

                    x = self._run_flex_attention(
                        query=padded_roped_query,
                        key=padded_roped_key,
                        value=padded_v,
                        block_mask=block_mask,
                        padded_length=padded_length,
                        x=x,
                        q=q,
                        k=k,
                        v=v,
                        roped_query=roped_query,
                        roped_key=roped_key,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        is_tf=is_tf,
                    )
                else:
                    x = self._run_flex_attention(
                        query=roped_query,
                        key=roped_key,
                        value=v,
                        block_mask=block_mask,
                        padded_length=0,
                        x=x,
                        q=q,
                        k=k,
                        v=v,
                        roped_query=roped_query,
                        roped_key=roped_key,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        is_tf=is_tf,
                    )
        else:
            try:
                frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
                ref_token_len = int(getattr(self, "_num_ref_tokens", 0) or 0)
                query_ref_token_len = int(getattr(self, "_query_ref_token_len", 0) or 0)
                video_token_len = q.shape[1] - query_ref_token_len
                num_video_frames = video_token_len // frame_seqlen if frame_seqlen > 0 else 0
                video_grid_sizes = grid_sizes.clone()
                video_grid_sizes[:, 0] = num_video_frames
                ref_info = {
                    "num_slots": int(getattr(self, "_ref_num_slots", 0) or 0),
                    "tokens_per_slot": int(getattr(self, "_ref_tokens_per_frame", 0) or 0),
                    "grid": getattr(self, "_ref_grid_sizes", None),
                }
                if "ref_token_len" not in kv_cache:
                    kv_cache["ref_token_len"] = torch.tensor(
                        [0], dtype=torch.long, device=q.device
                    )
                kv_cache["ref_token_len"].fill_(ref_token_len)
                sink_tokens = ref_token_len + self.sink_size * frame_seqlen
                kv_cache_size = kv_cache["k"].shape[1]
                cache_current_start = current_start + ref_token_len
                if query_ref_token_len > 0:
                    cache_current_start = 0
                cache_current_end = ref_token_len + current_start + video_token_len

                if self.use_relative_rope:
                    fast_rel = _REL_ROPE_CACHE_ENABLED and int(self.sink_size) == 0
                    if "k_raw" not in kv_cache or kv_cache["k_raw"].shape != kv_cache["k"].shape:
                        kv_cache["k_raw"] = torch.empty_like(kv_cache["k"])
                        if fast_rel:
                            kv_cache["k_roped"] = torch.empty_like(kv_cache["k"])
                            kv_cache["rel_rope_base_frame"] = 0
                    if fast_rel and "k_roped" not in kv_cache:
                        kv_cache["k_roped"] = torch.empty_like(kv_cache["k"])
                        kv_cache["rel_rope_base_frame"] = 0
                    # Reset the rope base at the start of a new stream (reset_stream
                    # zeroes global_end_index but reuses the cache tensors).
                    if fast_rel and int(kv_cache["global_end_index"].item()) == 0:
                        kv_cache["rel_rope_base_frame"] = 0

                    if self.local_attn_size != -1 and (cache_current_end > kv_cache["global_end_index"].item()) and (
                            video_token_len + kv_cache["local_end_index"].item() > kv_cache_size):
                        num_evicted_tokens = video_token_len + kv_cache["local_end_index"].item() - kv_cache_size
                        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                        if num_evicted_tokens < 0 or num_rolled_tokens < 0:
                            _dbg_print(
                                "relative_kv_cache.roll.bad_sizes",
                                current_start=current_start,
                                current_end=cache_current_end,
                                frame_seqlen=frame_seqlen,
                                ref_token_len=ref_token_len,
                                query_ref_token_len=query_ref_token_len,
                                sink_tokens=sink_tokens,
                                kv_cache_size=kv_cache_size,
                                video_token_len=video_token_len,
                                num_evicted_tokens=num_evicted_tokens,
                                num_rolled_tokens=num_rolled_tokens,
                            )
                            raise RuntimeError(
                                f"Invalid relative KV cache roll sizes: "
                                f"evict={num_evicted_tokens}, roll={num_rolled_tokens}"
                            )

                        kv_cache["k_raw"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["k_raw"][:,
                            sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["v"][:,
                            sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        if fast_rel:
                            kv_cache["k_roped"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache["k_roped"][:,
                                sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                        local_end_index = kv_cache["local_end_index"].item() + cache_current_end - \
                                          kv_cache["global_end_index"].item() - num_evicted_tokens
                    else:
                        local_end_index = kv_cache["local_end_index"].item() + cache_current_end - kv_cache[
                            "global_end_index"].item()
                    local_start_index = local_end_index - video_token_len

                    if local_start_index < sink_tokens or local_end_index > kv_cache["k_raw"].shape[1]:
                        _dbg_print(
                            "relative_kv_cache.index.out_of_range",
                            current_start=current_start,
                            current_end=cache_current_end,
                            frame_seqlen=frame_seqlen,
                            ref_token_len=ref_token_len,
                            query_ref_token_len=query_ref_token_len,
                            sink_tokens=sink_tokens,
                            kv_cache_size=kv_cache_size,
                            video_token_len=video_token_len,
                            local_start_index=local_start_index,
                            local_end_index=local_end_index,
                            kv_cache_k_raw=kv_cache["k_raw"],
                            kv_cache_v=kv_cache["v"],
                            global_end_index=kv_cache["global_end_index"],
                            local_end_index_tensor=kv_cache["local_end_index"],
                        )
                        raise RuntimeError(
                            f"Relative KV cache write out of range: "
                            f"[{local_start_index}:{local_end_index}] vs cache_len={kv_cache['k_raw'].shape[1]}"
                        )

                    if query_ref_token_len > 0:
                        kv_cache["k_raw"][:, :ref_token_len] = k[:, :query_ref_token_len].detach()
                        kv_cache["v"][:, :ref_token_len] = v[:, :query_ref_token_len]
                    kv_cache["k_raw"][:, local_start_index:local_end_index] = k[:, query_ref_token_len:].detach()
                    kv_cache["v"][:, local_start_index:local_end_index] = v[:, query_ref_token_len:]

                    max_attention_tokens = (
                        local_end_index - sink_tokens
                        if self.local_attn_size == -1
                        else int(self.local_attn_size) * frame_seqlen
                    )
                    recent_start = max(sink_tokens, local_end_index - max_attention_tokens)
                    visible_recent_tokens = local_end_index - recent_start
                    misalign = visible_recent_tokens % frame_seqlen
                    if misalign:
                        recent_start += misalign
                        visible_recent_tokens = local_end_index - recent_start

                    protected_video_tokens = max(0, sink_tokens - ref_token_len)

                    if not fast_rel:
                        video_key_parts = []
                        video_value_parts = []
                        if protected_video_tokens > 0:
                            video_key_parts.append(kv_cache["k_raw"][:, ref_token_len:sink_tokens])
                            video_value_parts.append(kv_cache["v"][:, ref_token_len:sink_tokens])
                        if visible_recent_tokens > 0:
                            video_key_parts.append(kv_cache["k_raw"][:, recent_start:local_end_index])
                            video_value_parts.append(kv_cache["v"][:, recent_start:local_end_index])
                        visible_video_raw = torch.cat(video_key_parts, dim=1) if video_key_parts else kv_cache["k_raw"][:, :0]
                        visible_video_v = torch.cat(video_value_parts, dim=1) if video_value_parts else kv_cache["v"][:, :0]
                        visible_video_frames = visible_video_raw.shape[1] // frame_seqlen if frame_seqlen > 0 else 0

                        attn_k_parts = []
                        attn_v_parts = []
                        query_parts = []
                        if ref_token_len > 0:
                            if ref_info["num_slots"] <= 0 or ref_info["tokens_per_slot"] <= 0 or ref_info["grid"] is None:
                                raise RuntimeError("Ref cache write requested but ref RoPE metadata is incomplete.")
                            ref_freqs = _build_ref_freqs(
                                freqs=freqs,
                                num_slots=ref_info["num_slots"],
                                tokens_per_slot=ref_info["tokens_per_slot"],
                                ref_grid=ref_info["grid"],
                                device=q.device,
                            )
                            attn_k_parts.append(
                                rope_apply_with_refimg(
                                    kv_cache["k_raw"][:, :ref_token_len],
                                    ref_freqs,
                                    self.num_heads,
                                ).type_as(v)
                            )
                            attn_v_parts.append(kv_cache["v"][:, :ref_token_len])
                            if query_ref_token_len > 0:
                                query_parts.append(
                                    rope_apply_with_refimg(
                                        q[:, :query_ref_token_len],
                                        ref_freqs,
                                        self.num_heads,
                                    ).type_as(v)
                                )

                        if visible_video_frames > 0:
                            visible_video_grid_sizes = grid_sizes.clone()
                            visible_video_grid_sizes[:, 0] = visible_video_frames
                            rel_k_frame_indices = torch.arange(
                                visible_video_frames, device=q.device, dtype=torch.long
                            )
                            attn_k_parts.append(
                                relative_rope_apply(
                                    visible_video_raw,
                                    visible_video_grid_sizes,
                                    freqs,
                                    frame_indices=rel_k_frame_indices,
                                ).type_as(v)
                            )
                            attn_v_parts.append(visible_video_v)
                            if num_video_frames <= visible_video_frames:
                                rel_q_frame_indices = rel_k_frame_indices[-num_video_frames:]
                            else:
                                rel_q_frame_indices = torch.arange(
                                    num_video_frames, device=q.device, dtype=torch.long
                                )
                            _rel_rope_debug_print(
                                "window_local",
                                once_key=("window_local", int(current_start)),
                                current_start=current_start,
                                frame_seqlen=frame_seqlen,
                                abs_frame_start=current_start // frame_seqlen if frame_seqlen > 0 else 0,
                                visible_video_frames=visible_video_frames,
                                num_video_frames=num_video_frames,
                                recent_start=recent_start,
                                local_start_index=local_start_index,
                                local_end_index=local_end_index,
                                rel_k_first=rel_k_frame_indices[0] if rel_k_frame_indices.numel() else -1,
                                rel_k_last=rel_k_frame_indices[-1] if rel_k_frame_indices.numel() else -1,
                                rel_q_first=rel_q_frame_indices[0] if rel_q_frame_indices.numel() else -1,
                                rel_q_last=rel_q_frame_indices[-1] if rel_q_frame_indices.numel() else -1,
                                ref_token_len=ref_token_len,
                            )
                            query_parts.append(
                                relative_rope_apply(
                                    q[:, query_ref_token_len:],
                                    video_grid_sizes,
                                    freqs,
                                    frame_indices=rel_q_frame_indices,
                                ).type_as(v)
                            )

                        roped_query = torch.cat(query_parts, dim=1) if len(query_parts) > 1 else query_parts[0]
                        attn_k = torch.cat(attn_k_parts, dim=1) if len(attn_k_parts) > 1 else attn_k_parts[0]
                        attn_v = torch.cat(attn_v_parts, dim=1) if len(attn_v_parts) > 1 else attn_v_parts[0]
                        x = attention(roped_query, attn_k, attn_v)
                    else:
                        # ── Fast relative path (opt E): cache post-RoPE video keys ──
                        # Rope only the new block's keys/query per step; the visible
                        # window's keys are already roped in kv_cache["k_roped"].
                        # Positions use an absolute counter (base = rel_rope_base_frame)
                        # whose differences match the window-local scheme; re-base
                        # (re-rope the window once) before nearing the rotary table.
                        visible_video_frames = (
                            visible_recent_tokens // frame_seqlen if frame_seqlen > 0 else 0
                        )
                        abs_frame_start = current_start // frame_seqlen
                        base_frame = int(kv_cache["rel_rope_base_frame"])
                        new_start_pos = abs_frame_start - base_frame

                        rope_table = int(freqs.shape[0])
                        rebase_limit = min(
                            _REL_ROPE_REBASE_MAX_POS,
                            rope_table - int(self.local_attn_size) - num_video_frames,
                        )
                        need_rebase = visible_video_frames > 0 and (
                            (new_start_pos + num_video_frames) > rebase_limit
                            or new_start_pos < 0
                        )
                        debug_new_start_pos_before_rebase = new_start_pos

                        if need_rebase:
                            # Re-base so the oldest visible frame maps to position 0,
                            # then re-rope the whole visible window once.
                            oldest_visible_abs = (
                                abs_frame_start + num_video_frames - visible_video_frames
                            )
                            base_frame = oldest_visible_abs
                            kv_cache["rel_rope_base_frame"] = base_frame
                            win_grid = grid_sizes.clone()
                            win_grid[:, 0] = visible_video_frames
                            kv_cache["k_roped"][:, recent_start:local_end_index] = causal_rope_apply(
                                kv_cache["k_raw"][:, recent_start:local_end_index],
                                win_grid,
                                freqs,
                                start_frame=0,
                            ).type_as(v)
                            new_start_pos = abs_frame_start - base_frame
                        else:
                            # Rope only the newly written block's keys.
                            kv_cache["k_roped"][:, local_start_index:local_end_index] = causal_rope_apply(
                                k[:, query_ref_token_len:],
                                video_grid_sizes,
                                freqs,
                                start_frame=new_start_pos,
                            ).type_as(v)

                        _rel_rope_debug_print(
                            "fast_cache",
                            once_key=("fast_cache", int(current_start)),
                            current_start=current_start,
                            frame_seqlen=frame_seqlen,
                            abs_frame_start=abs_frame_start,
                            base_frame=base_frame,
                            new_start_pos=new_start_pos,
                            temporal_index_start=new_start_pos,
                            temporal_index_end=new_start_pos + num_video_frames - 1,
                            new_start_pos_before_rebase=debug_new_start_pos_before_rebase,
                            num_video_frames=num_video_frames,
                            visible_video_frames=visible_video_frames,
                            recent_start=recent_start,
                            local_start_index=local_start_index,
                            local_end_index=local_end_index,
                            ref_token_len=ref_token_len,
                            rebase_limit=rebase_limit,
                            need_rebase=need_rebase,
                            rope_table=rope_table,
                        )

                        roped_video_query = causal_rope_apply(
                            q[:, query_ref_token_len:],
                            video_grid_sizes,
                            freqs,
                            start_frame=new_start_pos,
                        ).type_as(v)

                        video_k = kv_cache["k_roped"][:, recent_start:local_end_index]
                        video_v = kv_cache["v"][:, recent_start:local_end_index]

                        if ref_token_len > 0:
                            if ref_info["num_slots"] <= 0 or ref_info["tokens_per_slot"] <= 0 or ref_info["grid"] is None:
                                raise RuntimeError("Ref cache write requested but ref RoPE metadata is incomplete.")
                            ref_freqs = _build_ref_freqs(
                                freqs=freqs,
                                num_slots=ref_info["num_slots"],
                                tokens_per_slot=ref_info["tokens_per_slot"],
                                ref_grid=ref_info["grid"],
                                device=q.device,
                            )
                            ref_k = rope_apply_with_refimg(
                                kv_cache["k_raw"][:, :ref_token_len],
                                ref_freqs,
                                self.num_heads,
                            ).type_as(v)
                            attn_k = torch.cat([ref_k, video_k], dim=1)
                            attn_v = torch.cat([kv_cache["v"][:, :ref_token_len], video_v], dim=1)
                            if query_ref_token_len > 0:
                                ref_q = rope_apply_with_refimg(
                                    q[:, :query_ref_token_len],
                                    ref_freqs,
                                    self.num_heads,
                                ).type_as(v)
                                roped_query = torch.cat([ref_q, roped_video_query], dim=1)
                            else:
                                roped_query = roped_video_query
                        else:
                            attn_k = video_k
                            attn_v = video_v
                            roped_query = roped_video_query

                        x = attention(roped_query, attn_k, attn_v)

                    kv_cache["global_end_index"].fill_(cache_current_end)
                    kv_cache["local_end_index"].fill_(local_end_index)
                else:
                    current_start_frame = current_start // frame_seqlen

                    if query_ref_token_len > 0:
                        if ref_info["num_slots"] <= 0 or ref_info["tokens_per_slot"] <= 0 or ref_info["grid"] is None:
                            raise RuntimeError("Ref cache write requested but ref RoPE metadata is incomplete.")
                        ref_freqs = _build_ref_freqs(
                            freqs=freqs,
                            num_slots=ref_info["num_slots"],
                            tokens_per_slot=ref_info["tokens_per_slot"],
                            ref_grid=ref_info["grid"],
                            device=q.device,
                        )
                        roped_query = torch.cat([
                            rope_apply_with_refimg(q[:, :query_ref_token_len], ref_freqs, self.num_heads),
                            causal_rope_apply(
                                q[:, query_ref_token_len:],
                                video_grid_sizes,
                                freqs,
                                start_frame=current_start_frame,
                            ),
                        ], dim=1).type_as(v)
                        roped_key = torch.cat([
                            rope_apply_with_refimg(k[:, :query_ref_token_len], ref_freqs, self.num_heads),
                            causal_rope_apply(
                                k[:, query_ref_token_len:],
                                video_grid_sizes,
                                freqs,
                                start_frame=current_start_frame,
                            ),
                        ], dim=1).type_as(v)
                    else:
                        roped_query = causal_rope_apply(
                            q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                        roped_key = causal_rope_apply(
                            k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

                    num_new_tokens = roped_query.shape[1]

                    if self.local_attn_size != -1 and (cache_current_end > kv_cache["global_end_index"].item()) and (
                            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                        if num_evicted_tokens < 0 or num_rolled_tokens < 0:
                            _dbg_print(
                                "kv_cache.roll.bad_sizes",
                                current_start=current_start,
                                current_end=cache_current_end,
                                current_start_frame=current_start_frame,
                                frame_seqlen=frame_seqlen,
                                ref_token_len=ref_token_len,
                                query_ref_token_len=query_ref_token_len,
                                sink_tokens=sink_tokens,
                                kv_cache_size=kv_cache_size,
                                num_new_tokens=num_new_tokens,
                                num_evicted_tokens=num_evicted_tokens,
                                num_rolled_tokens=num_rolled_tokens,
                                kv_cache_k=kv_cache["k"],
                                kv_cache_v=kv_cache["v"],
                                global_end_index=kv_cache["global_end_index"],
                                local_end_index_tensor=kv_cache["local_end_index"],
                                grid_sizes=grid_sizes,
                            )
                            raise RuntimeError(
                                f"Invalid KV cache roll sizes: evict={num_evicted_tokens}, roll={num_rolled_tokens}"
                            )

                        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["k"][:,
                            sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["v"][:,
                            sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                        local_end_index = kv_cache["local_end_index"].item() + cache_current_end - \
                                          kv_cache["global_end_index"].item() - num_evicted_tokens
                        local_start_index = local_end_index - num_new_tokens
                    else:
                        local_end_index = kv_cache["local_end_index"].item() + cache_current_end - kv_cache[
                            "global_end_index"].item()
                        local_start_index = local_end_index - num_new_tokens

                    # 只在即将越界时打印；正常路径无输出。
                    if local_start_index < 0 or local_end_index > kv_cache["k"].shape[1]:
                        _dbg_print(
                            "kv_cache.index.out_of_range",
                            current_start=current_start,
                            current_end=cache_current_end,
                            current_start_frame=current_start_frame,
                            frame_seqlen=frame_seqlen,
                            ref_token_len=ref_token_len,
                            query_ref_token_len=query_ref_token_len,
                            sink_tokens=sink_tokens,
                            kv_cache_size=kv_cache_size,
                            num_new_tokens=num_new_tokens,
                            cache_current_start=cache_current_start,
                            kv_cache_k=kv_cache["k"],
                            kv_cache_v=kv_cache["v"],
                            roped_key=roped_key,
                            value=v,
                            global_end_index=kv_cache["global_end_index"],
                            local_end_index_tensor=kv_cache["local_end_index"],
                            grid_sizes=grid_sizes,
                        )
                        raise RuntimeError(
                            f"KV cache write out of range: "
                            f"[{local_start_index}:{local_end_index}] vs cache_len={kv_cache['k'].shape[1]}"
                        )

                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key.detach()
                    kv_cache["v"][:, local_start_index:local_end_index] = v

                    attn_start = max(sink_tokens, local_end_index - self.max_attention_size)
                    if sink_tokens > 0 and attn_start > sink_tokens:
                        attn_k = torch.cat([
                            kv_cache["k"][:, :sink_tokens],
                            kv_cache["k"][:, attn_start:local_end_index],
                        ], dim=1)
                        attn_v = torch.cat([
                            kv_cache["v"][:, :sink_tokens],
                            kv_cache["v"][:, attn_start:local_end_index],
                        ], dim=1)
                    else:
                        attn_k = kv_cache["k"][:, :local_end_index]
                        attn_v = kv_cache["v"][:, :local_end_index]
                    x = attention(roped_query, attn_k, attn_v)
                    kv_cache["global_end_index"].fill_(cache_current_end)
                    kv_cache["local_end_index"].fill_(local_end_index)
            except Exception as e:
                if _is_checkpoint_stop_signal(e):
                    raise
                _dbg_print(
                    "self_attn.kv_cache_path.failed",
                    error=repr(e),
                    x=x,
                    q=q,
                    k=k,
                    v=v,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    current_start=current_start,
                    cache_start=cache_start,
                    kv_cache_k=kv_cache.get("k", None) if isinstance(kv_cache, dict) else None,
                    kv_cache_v=kv_cache.get("v", None) if isinstance(kv_cache, dict) else None,
                    global_end_index=kv_cache.get("global_end_index", None) if isinstance(kv_cache, dict) else None,
                    local_end_index_tensor=kv_cache.get("local_end_index", None) if isinstance(kv_cache,
                                                                                               dict) else None,
                )
                raise

        x = x.flatten(2)
        try:
            x = self.o(x)
        except Exception as e:
            if _is_checkpoint_stop_signal(e):
                raise
            _dbg_print(
                "self_attn.output_proj.failed",
                error=repr(e),
                x=x,
            )
            raise
        return x


# =========================================================
# Cross-Attention
# =========================================================

class CausalWanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = attention(q, k, v, k_lens=context_lens, backend='flash_attn')
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanI2VCrossAttention(WanI2VCrossAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
                v_img = self.v_img(context_img).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
                crossattn_cache["k_img"] = k_img
                crossattn_cache["v_img"] = v_img
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
                k_img = crossattn_cache["k_img"]
                v_img = crossattn_cache["v_img"]
        else:
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        img_x = attention(q, k_img, v_img, k_lens=None, backend='flash_attn')
        x = attention(q, k, v, k_lens=context_lens, backend='flash_attn')

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


# =========================================================
# Attention Block
# =========================================================

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': CausalWanCrossAttention,
    'i2v_cross_attn': CausalWanI2VCrossAttention,
}


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 cross_attn_type="t2v_cross_attn",
                 use_relative_rope=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
            use_relative_rope=use_relative_rope,
        )
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
            self,
            x,  # [B, L, C]
            e,  # [B, L_frame, 6, C]
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask: BlockMask | None = None,
            kv_cache: dict | None = None,
            crossattn_cache=None,
            current_start: int = 0,
            cache_start: int | None = None,
    ):
        token_level_modulation = e.shape[1] == x.shape[1]
        num_frames = e.shape[1]
        if (not token_level_modulation) and x.shape[1] % num_frames != 0:
            _dbg_print(
                "attention_block.bad_frame_token_split",
                x=x,
                e=e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                current_start=current_start,
                cache_start=cache_start,
            )
            raise RuntimeError(
                f"x token length {x.shape[1]} is not divisible by e frames {num_frames}"
            )
        if token_level_modulation:
            num_frames, frame_seqlen = x.shape[1], 1
        else:
            frame_seqlen = x.shape[1] // num_frames

        try:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

            def modulate_norm1(value):
                normed = self.norm1(value)
                if token_level_modulation:
                    return normed * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
                return (
                    normed.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
                ).flatten(1, 2)

            def modulate_norm2(value):
                normed = self.norm2(value)
                if token_level_modulation:
                    return normed * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
                return (
                    normed.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]
                ).flatten(1, 2)

            def apply_gate(value, gate):
                if token_level_modulation:
                    return value * gate.squeeze(2)
                return (value.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(1, 2)

            y = self.self_attn(
                modulate_norm1(x),
                seq_lens,
                grid_sizes,
                freqs,
                block_mask=block_mask,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=cache_start,
            )
            x = x + apply_gate(y, e[2])

            def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
                x = x + self.cross_attn(self.norm3(x), context,
                                        context_lens, crossattn_cache=crossattn_cache)
                y = self.ffn(modulate_norm2(x))
                x = x + apply_gate(y, e[5])
                return x

            x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
            return x
        except Exception as err:
            if _is_checkpoint_stop_signal(err):
                raise
            _dbg_print(
                "attention_block.forward.failed",
                error=repr(err),
                x=x,
                e0=e[0] if isinstance(e, tuple) else e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                context=context,
                context_lens=context_lens,
                block_mask=_dbg_block_mask(block_mask),
                kv_cache_is_none=(kv_cache is None),
                crossattn_cache_is_none=(crossattn_cache is None),
                current_start=current_start,
                cache_start=cache_start,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
            )
            raise


# ===== Causal Head：沿用 2.2 的形状，只把 e 理解为 [B, L, C] =====

class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim_ = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        """
        和 2.2 一样：
        x: [B, L_token, C]
        e: [B, F, 1, C] or [B, F, C] before unsqueeze
        """
        num_frames = e.shape[1]
        if x.shape[1] % num_frames != 0:
            _dbg_print(
                "head.bad_frame_token_split",
                x=x,
                e=e,
                num_frames=num_frames,
            )
            raise RuntimeError(
                f"Head split failed: x_len={x.shape[1]}, e_frames={num_frames}"
            )
        frame_seqlen = x.shape[1] // num_frames

        try:
            e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
            x = self.head(
                self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            )
            return x
        except Exception as err:
            if _is_checkpoint_stop_signal(err):
                raise
            _dbg_print(
                "head.forward.failed",
                error=repr(err),
                x=x,
                e=e[0] if isinstance(e, tuple) else e,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
            )
            raise


# ===== CausalWanModel 2.2 =====

class CausalWanModel(ModelMixin, ConfigMixin):
    """
    基于 Wan 2.2 结构，加入：
    - causal / blockwise / local attention
    - flex_attention + BlockMask 训练
    - KV cache 推理
    - teacher forcing (通过 clean_x / aug_t)
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 local_attn_size=-1,
                 num_frame_per_block=3,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 act_control_in_dim=32,
                 use_relative_rope=False,
                 downscale_factor_control_adapter=8):
        super().__init__()
        print(f"Initializing CausalWanModel with model_type={model_type}, use_relative_rope={use_relative_rope}")
        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v', 'ci2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_relative_rope = bool(use_relative_rope)

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )

        # cross_attn_type = 'i2v_cross_attn' if model_type in ['i2v', 'ti2v'] else 't2v_cross_attn'
        cross_attn_type = 'i2v_cross_attn' if model_type in ['i2v'] else 't2v_cross_attn'

        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                dim, ffn_dim, num_heads,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                cross_attn_type=cross_attn_type,
                use_relative_rope=use_relative_rope,
            )
            for _ in range(num_layers)
        ])

        self.head = CausalHead(dim, out_dim, patch_size, eps)
        self.act_control_adapter = SimpleAdapter(
            act_control_in_dim, self.dim,
            kernel_size=self.patch_size[1:], stride=self.patch_size[1:],
            downscale_factor=downscale_factor_control_adapter)
        self.act_control_adapter.requires_grad_(False)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask: BlockMask | None = None
        self._block_mask_cache_key = None
        self._block_mask_cache = {}
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value: bool = False):
        self.gradient_checkpointing = value

    @staticmethod
    def _frame_block_token_ranges(
            device: torch.device | str,
            num_frames: int,
            frame_seqlen: int,
            num_frame_per_block=1,
            independent_first_frame=False,
    ):
        if independent_first_frame and num_frames > 0:
            starts = [0]
            ends = [frame_seqlen]
            for frame_start in range(1, num_frames, num_frame_per_block):
                frame_end = min(frame_start + num_frame_per_block, num_frames)
                starts.append(frame_start * frame_seqlen)
                ends.append(frame_end * frame_seqlen)
        else:
            starts = list(range(0, num_frames * frame_seqlen, frame_seqlen * num_frame_per_block))
            ends = [
                min(start + frame_seqlen * num_frame_per_block, num_frames * frame_seqlen)
                for start in starts
            ]

        return (
            torch.tensor(starts, device=device, dtype=torch.long),
            torch.tensor(ends, device=device, dtype=torch.long),
        )

    @staticmethod
    def _build_block_mask_from_visibility(
            block_visibility: torch.Tensor,
            mask_mod,
            seq_length: int,
            device: torch.device | str,
            block_size: int = 128,
    ) -> BlockMask:
        """Create a BlockMask from block-level visibility without materializing a dense token mask."""
        dense_blocks = block_visibility.unsqueeze(0).unsqueeze(0).to(dtype=torch.int32)
        kv_num_blocks = dense_blocks.sum(dim=-1).to(torch.int32, memory_format=torch.contiguous_format)
        kv_indices = torch.argsort(dense_blocks, dim=-1, descending=True, stable=True).to(
            torch.int32, memory_format=torch.contiguous_format
        )
        return BlockMask.from_kv_blocks(
            kv_num_blocks.to(device),
            kv_indices.to(device),
            full_kv_num_blocks=None,
            full_kv_indices=None,
            BLOCK_SIZE=(block_size, block_size),
            mask_mod=mask_mod,
            seq_lengths=(seq_length, seq_length),
        )

    @staticmethod
    def _blockwise_causal_visibility(
            total_padded: int,
            total_length: int,
            ref_token_len: int,
            video_length: int,
            block_starts: torch.Tensor,
            block_ends: torch.Tensor,
            frame_seqlen: int,
            local_attn_size=-1,
            block_size: int = 128,
    ) -> torch.Tensor:
        """Build only [query_block, key_block] visibility, avoiding seq_len^2 dense masks."""
        q_blocks = math.ceil(total_padded / block_size)
        block_visibility = torch.zeros((q_blocks, q_blocks), dtype=torch.bool)
        ref_token_len = int(ref_token_len)
        video_offset = ref_token_len
        video_end = ref_token_len + int(video_length)

        def mark_token_interval(row: int, start: int, end: int):
            start = max(0, min(int(start), total_padded))
            end = max(0, min(int(end), total_padded))
            if start >= end:
                return
            block_start = start // block_size
            block_end = (end - 1) // block_size + 1
            block_visibility[row, block_start:block_end] = True

        frame_ranges = [(int(s), int(e)) for s, e in zip(block_starts.tolist(), block_ends.tolist())]

        for q_block in range(q_blocks):
            q_start = q_block * block_size
            q_end = min((q_block + 1) * block_size, total_padded)
            block_visibility[q_block, q_block] = True

            q_has_ref = ref_token_len > 0 and q_start < ref_token_len and q_end > 0
            if q_has_ref:
                mark_token_interval(q_block, 0, ref_token_len)

            q_video_start = max(q_start, video_offset)
            q_video_end = min(q_end, video_end)
            if q_video_start >= q_video_end:
                continue

            if ref_token_len > 0:
                mark_token_interval(q_block, 0, ref_token_len)

            q_video_start -= video_offset
            q_video_end -= video_offset
            for frame_block_start, frame_block_end in frame_ranges:
                if max(q_video_start, frame_block_start) >= min(q_video_end, frame_block_end):
                    continue
                if local_attn_size == -1:
                    visible_start = 0
                else:
                    visible_start = max(0, frame_block_end - int(local_attn_size) * int(frame_seqlen))
                mark_token_interval(q_block, video_offset + visible_start, video_offset + frame_block_end)

        return block_visibility

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
            device: torch.device | str,
            num_frames: int,
            frame_seqlen: int,
            num_frame_per_block=1,
            local_attn_size=-1,
            independent_first_frame=False,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        block_starts, block_ends = CausalWanModel._frame_block_token_ranges(
            device=device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
        )
        for start, end in zip(block_starts, block_ends):
            ends[start:end] = end

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                        (kv_idx < ends[q_idx])
                        & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_visibility = CausalWanModel._blockwise_causal_visibility(
            total_padded=total_length + padded_length,
            total_length=total_length,
            ref_token_len=0,
            video_length=total_length,
            block_starts=block_starts,
            block_ends=block_ends,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
        )
        return CausalWanModel._build_block_mask_from_visibility(
            block_visibility,
            mask_mod=attention_mask,
            seq_length=total_length + padded_length,
            device=device,
        )

    @staticmethod
    def _prepare_teacher_forcing_mask(
            device: torch.device | str,
            num_frames: int,
            frame_seqlen: int,
            num_frame_per_block=1,
            independent_first_frame=False,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen

        context_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_context_starts = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_context_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_noise_starts = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        noise_noise_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        block_starts, block_ends = CausalWanModel._frame_block_token_ranges(
            device=device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
        )

        for start, end in zip(block_starts, block_ends):
            clean_end = torch.minimum(end, end.new_tensor(clean_ends))
            context_ends[start:end] = clean_end

        for clean_start, clean_end in zip(block_starts, block_ends):
            start = clean_ends + clean_start
            end = clean_ends + clean_end
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = clean_start

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])

            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = (q_idx == kv_idx)
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_with_ref(
            device: torch.device | str,
            num_frames: int,
            frame_seqlen: int,
            ref_token_len: int,
            num_frame_per_block=1,
            local_attn_size=-1,
            independent_first_frame=False,
    ) -> BlockMask:
        video_length = num_frames * frame_seqlen
        total_length = int(ref_token_len) + video_length
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(video_length + padded_length, device=device, dtype=torch.long)
        block_starts, block_ends = CausalWanModel._frame_block_token_ranges(
            device=device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
        )
        for start, end in zip(block_starts, block_ends):
            ends[start:end] = end

        def attention_mask(b, h, q_idx, kv_idx):
            q_is_ref = q_idx < ref_token_len
            kv_is_ref = kv_idx < ref_token_len
            q_video = torch.clamp(q_idx - ref_token_len, min=0, max=max(video_length - 1, 0))
            kv_video = torch.clamp(kv_idx - ref_token_len, min=0, max=max(video_length - 1, 0))
            if local_attn_size == -1:
                video_visible = (kv_video < ends[q_video]) | (q_video == kv_video)
            else:
                video_visible = (
                    (kv_video < ends[q_video])
                    & (kv_video >= (ends[q_video] - local_attn_size * frame_seqlen))
                ) | (q_video == kv_video)
            video_mask = (q_idx >= ref_token_len) & (q_idx < total_length) & (
                kv_is_ref | ((kv_idx >= ref_token_len) & (kv_idx < total_length) & video_visible)
            )
            return (q_is_ref & kv_is_ref) | video_mask | (q_idx == kv_idx)

        block_visibility = CausalWanModel._blockwise_causal_visibility(
            total_padded=total_length + padded_length,
            total_length=total_length,
            ref_token_len=ref_token_len,
            video_length=video_length,
            block_starts=block_starts,
            block_ends=block_ends,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
        )
        return CausalWanModel._build_block_mask_from_visibility(
            block_visibility,
            mask_mod=attention_mask,
            seq_length=total_length + padded_length,
            device=device,
        )

    @staticmethod
    def _prepare_teacher_forcing_mask_with_ref_i2v(
            device: torch.device | str,
            num_frames: int,
            frame_seqlen: int,
            ref_token_len: int,
            num_frame_per_block=1,
            independent_first_frame=False,
    ) -> BlockMask:
        branch_len = num_frames * frame_seqlen
        total_length = int(ref_token_len) + branch_len * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        total_padded = total_length + padded_length
        clean_offset = int(ref_token_len)
        noisy_offset = clean_offset + branch_len

        context_ends = torch.zeros(total_padded, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_padded, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_padded, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_padded, device=device, dtype=torch.long)

        block_starts, block_ends = CausalWanModel._frame_block_token_ranges(
            device=device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
        )
        for start, end in zip(block_starts, block_ends):
            context_ends[clean_offset + start:clean_offset + end] = clean_offset + end
            noise_start = noisy_offset + start
            noise_end = noisy_offset + end
            noise_noise_starts[noise_start:noise_end] = noise_start
            noise_noise_ends[noise_start:noise_end] = noise_end
            noise_context_ends[noise_start:noise_end] = clean_offset + start

        def attention_mask(b, h, q_idx, kv_idx):
            q_is_ref = q_idx < ref_token_len
            kv_is_ref = kv_idx < ref_token_len
            clean_q = (q_idx >= clean_offset) & (q_idx < noisy_offset)
            noisy_q = (q_idx >= noisy_offset) & (q_idx < total_length)
            clean_mask = clean_q & (kv_idx >= clean_offset) & (kv_idx < context_ends[q_idx])
            noisy_self = noisy_q & (kv_idx >= noise_noise_starts[q_idx]) & (kv_idx < noise_noise_ends[q_idx])
            noisy_history = noisy_q & (kv_idx >= clean_offset) & (kv_idx < noise_context_ends[q_idx])
            return (q_is_ref & kv_is_ref) | ((~q_is_ref) & (kv_is_ref | clean_mask | noisy_self | noisy_history)) | (q_idx == kv_idx)

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_padded,
            KV_LEN=total_padded,
            _compile=False,
            device=device,
        )

    def _maybe_build_block_mask(
            self,
            device,
            num_frames,
            frame_seqlen,
            is_teacher_forcing,
            ref_token_len=0,
            independent_first_frame=None,
    ):
        """
        只有 mask 配置变化时重建，避免不同帧数/分辨率/TF 状态复用旧 BlockMask。
        正常不打印；只有后续 flex_attention 崩时会打印 mask repr 和 shape 上下文。
        """
        if independent_first_frame is None:
            independent_first_frame = self.independent_first_frame
        cache_key = (
            str(device),
            int(num_frames),
            int(frame_seqlen),
            int(self.num_frame_per_block),
            int(self.local_attn_size),
            bool(is_teacher_forcing),
            bool(independent_first_frame),
            int(ref_token_len),
        )
        cached_mask = self._block_mask_cache.get(cache_key)
        if cached_mask is not None:
            self.block_mask = cached_mask
            self._block_mask_cache_key = cache_key
            return

        if is_teacher_forcing and ref_token_len > 0:
            block_mask = self._prepare_teacher_forcing_mask_with_ref_i2v(
                device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                ref_token_len=ref_token_len,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=bool(independent_first_frame),
            )
        elif is_teacher_forcing:
            block_mask = self._prepare_teacher_forcing_mask(
                device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=bool(independent_first_frame),
            )
        elif ref_token_len > 0:
            block_mask = self._prepare_blockwise_causal_attn_mask_with_ref(
                device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                ref_token_len=ref_token_len,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                independent_first_frame=bool(independent_first_frame),
            )
        else:
            block_mask = self._prepare_blockwise_causal_attn_mask(
                device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                independent_first_frame=bool(independent_first_frame),
            )
        self.block_mask = block_mask
        self._block_mask_cache_key = cache_key
        self._block_mask_cache[cache_key] = block_mask

    def _apply_control_adapters(self, x,  act_context=None, act_context_scale=1.0):
        """复用action control adapter 逻辑；异常时打印 shape。"""
        try:
            y_action = None

            if act_context is not None and hasattr(self,
                                                   "act_control_adapter") and self.act_control_adapter is not None:
                x_new = []
                y_action = [self.act_control_adapter(u.unsqueeze(0)) for u in act_context]
                for u, v in zip(x, y_action):
                    t_f = u.shape[2]
                    c_f = v.shape[2]
                    if t_f > c_f:
                        offset = t_f - c_f
                        u = torch.cat([u[:, :, :offset], u[:, :, offset:] + v * act_context_scale], dim=2)
                    else:
                        u = u + v * act_context_scale
                    x_new.append(u)
                x = x_new
            return x,  y_action
        except Exception as e:
            if _is_checkpoint_stop_signal(e):
                raise
            _dbg_print(
                "control_adapter.failed",
                error=repr(e),
                act_context=act_context,
                act_context_scale=act_context_scale,
            )
            raise

    def _prepare_ref_tokens(self, ref_latents=None, ref_mask=None, batch_size=None, device=None, dtype=None):
        if ref_latents is None:
            return None
        ref_latents = ref_latents.detach()
        in_ch_need = self.patch_embedding.in_channels
        if ref_latents.ndim == 4:
            ref_latents = ref_latents.unsqueeze(0).unsqueeze(3)
        elif ref_latents.ndim == 5 and ref_latents.shape[2] == in_ch_need:
            ref_latents = ref_latents.unsqueeze(3)
        elif ref_latents.ndim == 5:
            ref_latents = ref_latents.unsqueeze(0)
        if ref_latents.ndim != 6:
            raise ValueError(
                f"ref_latents must be [B,K,C,T,H,W], [K,C,T,H,W], or [K,C,H,W], got {tuple(ref_latents.shape)}"
            )
        if device is not None or dtype is not None:
            ref_latents = ref_latents.to(device=device, dtype=dtype)
        B, K, C, T, H, W = ref_latents.shape
        if batch_size is not None and B == 1 and batch_size != 1:
            ref_latents = ref_latents.expand(batch_size, -1, -1, -1, -1, -1)
            B = batch_size
        elif batch_size is not None and B != batch_size:
            raise ValueError(f"ref_latents batch size {B} does not match video batch size {batch_size}")

        if ref_mask is not None:
            ref_mask = ref_mask.detach()
            if device is not None or dtype is not None:
                ref_mask = ref_mask.to(device=device, dtype=dtype)
            if ref_mask.ndim == 1:
                ref_mask = ref_mask.unsqueeze(0)
            if ref_mask.shape[0] == 1 and B != 1:
                ref_mask = ref_mask.expand(B, -1)
            if ref_mask.shape[:2] != (B, K):
                raise ValueError(f"ref_mask shape {tuple(ref_mask.shape)} does not match ref slots {(B, K)}")
        else:
            ref_mask = ref_latents.new_ones(B, K)

        ref_flat = ref_latents.reshape(B * K, C, T, H, W)
        if C < in_ch_need:
            pad = ref_flat.new_zeros(B * K, in_ch_need - C, T, H, W)
            ref_flat = torch.cat([ref_flat, pad], dim=1)
        elif C > in_ch_need:
            ref_flat = ref_flat[:, :in_ch_need]
        ref_features = self.patch_embedding(ref_flat)
        _, _, patch_t, patch_h, patch_w = ref_features.shape
        tokens_per_slot = patch_t * patch_h * patch_w
        ref_tokens = ref_features.flatten(2).transpose(1, 2)
        ref_tokens = ref_tokens.reshape(B, K, tokens_per_slot, self.dim)
        ref_tokens = ref_tokens * ref_mask[:, :, None, None].to(dtype=ref_tokens.dtype)
        return {
            "tokens": ref_tokens.reshape(B, K * tokens_per_slot, self.dim),
            "num_slots": int(K),
            "tokens_per_slot": int(tokens_per_slot),
            "token_len": int(K * tokens_per_slot),
            "grid": (int(patch_t), int(patch_h), int(patch_w)),
        }

    def _estimate_ref_token_len(self, ref_latents=None):
        if ref_latents is None:
            return 0

        in_ch_need = self.patch_embedding.in_channels
        if ref_latents.ndim == 4:
            k, _, h, w = ref_latents.shape
            t = 1
        elif ref_latents.ndim == 5 and ref_latents.shape[2] == in_ch_need:
            _, k, _, h, w = ref_latents.shape
            t = 1
        elif ref_latents.ndim == 5:
            k, _, t, h, w = ref_latents.shape
        elif ref_latents.ndim == 6:
            _, k, _, t, h, w = ref_latents.shape
        else:
            return 0

        patch_t, patch_h, patch_w = self.patch_size
        return int(k) * (int(t) // int(patch_t)) * (int(h) // int(patch_h)) * (int(w) // int(patch_w))

    def _prepend_ref_tokens(self, x, e=None, ref_info=None, *args, **kwargs):
        return x, e

    def _expand_frame_modulation_to_tokens(self, e0, frame_seqlen):
        if e0.shape[1] == 0:
            return e0
        return e0.repeat_interleave(int(frame_seqlen), dim=1)

    def _zero_ref_modulation(self, batch_size, token_len, device, dtype):
        ref_t = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        ref_e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, ref_t.flatten()).to(dtype)
        )
        ref_e0 = self.time_projection(ref_e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=ref_t.shape)
        return ref_e0.expand(-1, int(token_len), -1, -1)

    # ==== 推理：逐帧/逐块，带 kv_cache ====

    def _forward_inference(
            self,
            x,
            t,
            context,
            seq_len,
            y=None,
            kv_cache: list[dict] | None = None,
            crossattn_cache: dict = None,
            current_start: int = 0,
            cache_start: int = 0,
            clip_fea=None,
            act_context=None,
            ref_latents=None,
            ref_mask=None,
            act_context_scale=1.0,
    ):
        if torch.is_grad_enabled():
            return self._forward_train(
                x=x,
                t=t,
                context=context,
                seq_len=seq_len,
                y=y,
                clip_fea=clip_fea,
                act_context=act_context,
                ref_latents=ref_latents,
                ref_mask=ref_mask,
                act_context_scale=act_context_scale,
            )

        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if seq_len is None:
            seq_len = x.shape[2] * x.shape[-2] * x.shape[-1] // (
                        self.patch_size[1] * self.patch_size[2] * self.patch_size[0])

        if y is not None and self.model_type in ['i2v', 'ti2v']:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x,  _ = self._apply_control_adapters(
            x,
            act_context=act_context,
            act_context_scale=act_context_scale,
        )

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x, dim=0)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                )
                for u in context
            ])
        )

        if clip_fea is not None and hasattr(self, "img_emb"):
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        ref_info = self._prepare_ref_tokens(
            ref_latents=ref_latents,
            ref_mask=ref_mask,
            batch_size=x.shape[0],
            device=device,
            dtype=x.dtype,
        )
        N_r = ref_info["token_len"] if ref_info is not None else 0
        include_ref_tokens = ref_info is not None and (kv_cache is None or current_start == 0)
        query_ref_token_len = N_r if include_ref_tokens else 0
        if include_ref_tokens:
            ref_tokens = ref_info["tokens"]
            e0 = self._expand_frame_modulation_to_tokens(e0, math.prod(grid_sizes[0][1:]).item())
            ref_e0 = self._zero_ref_modulation(
                batch_size=x.shape[0],
                token_len=query_ref_token_len,
                device=device,
                dtype=x.dtype,
            )
            x = torch.cat([ref_tokens, x], dim=1)
            e0 = torch.cat([ref_e0, e0], dim=1)

        for block in self.blocks:
            block.self_attn._is_teacher_forcing = False
            block.self_attn._num_ref_tokens = N_r
            block.self_attn._query_ref_token_len = query_ref_token_len
            if ref_info is not None:
                block.self_attn._ref_num_slots = ref_info["num_slots"]
                block.self_attn._ref_tokens_per_frame = ref_info["tokens_per_slot"]
                block.self_attn._ref_grid_sizes = ref_info["grid"]
            else:
                block.self_attn._ref_num_slots = 0
                block.self_attn._ref_tokens_per_frame = None
                block.self_attn._ref_grid_sizes = None

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kw):
                return module(*inputs, **kw)

            return custom_forward

        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    dict(
                        kv_cache=kv_cache[idx],
                        current_start=current_start,
                        cache_start=cache_start,
                    )
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs, use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[idx],
                        "crossattn_cache": crossattn_cache[idx] if crossattn_cache is not None else None,
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        if query_ref_token_len > 0:
            x = x[:, query_ref_token_len:]
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    # ==== 训练：flex_attention + BlockMask，支持 teacher forcing ====

    def _forward_train(
            self,
            x,
            t,
            context,
            seq_len,
            y=None,
            clean_x=None,
            aug_t=None,
            clip_fea=None,
            act_context=None,
            ref_latents=None,
            ref_mask=None,
            act_context_scale=1.0,
            current_start: int = 0,
    ):
        try:
            if self.model_type == 'i2v':
                assert y is not None
            device = self.patch_embedding.weight.device
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if seq_len is None:
                seq_len = x.shape[2] * x.shape[-2] * x.shape[-1] // (
                            self.patch_size[1] * self.patch_size[2] * self.patch_size[0])

            num_frames = x.shape[2]
            frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
            print("_forward_train: x[0]", x[0].shape)
            absolute_start_frame = int(current_start) // int(frame_seqlen) if frame_seqlen > 0 else 0
            mask_independent_first_frame = bool(self.independent_first_frame) and absolute_start_frame == 0
            ref_token_len = self._estimate_ref_token_len(ref_latents)
            self._maybe_build_block_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                is_teacher_forcing=(clean_x is not None),
                ref_token_len=ref_token_len,
                independent_first_frame=mask_independent_first_frame,
            )
            if y is not None and self.model_type in ['i2v', 'ti2v']:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            x, y_action = self._apply_control_adapters(
                x,
                act_context=act_context,
                act_context_scale=act_context_scale,
            )

            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
            )
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
            assert seq_lens.max() <= seq_len

            max_len = seq_lens[0].item()
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, max_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ])

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ])
            )

            if clip_fea is not None and hasattr(self, "img_emb"):
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)

            ref_info = self._prepare_ref_tokens(
                ref_latents=ref_latents,
                ref_mask=ref_mask,
                batch_size=x.shape[0],
                device=device,
                dtype=x.dtype,
            )
            N_r = ref_info["token_len"] if ref_info is not None else 0
            if N_r != ref_token_len:
                ref_token_len = N_r
                self._maybe_build_block_mask(
                    device=device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    is_teacher_forcing=(clean_x is not None),
                    ref_token_len=ref_token_len,
                    independent_first_frame=mask_independent_first_frame,
                )

            if clean_x is not None:
                if y is not None and self.model_type in ['i2v', 'ti2v']:
                    clean_x = [torch.cat([u, v], dim=0) for u, v in zip(clean_x, y)]

                clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]



                if act_context is not None and hasattr(self,
                                                       "act_control_adapter") and self.act_control_adapter is not None:
                    x_new = []
                    for u, v in zip(clean_x, y_action):
                        t_f = u.shape[2]
                        c_f = v.shape[2]
                        if t_f > c_f:
                            offset = t_f - c_f
                            u = torch.cat([u[:, :, :offset], u[:, :, offset:] + v * act_context_scale], dim=2)
                        else:
                            u = u + v * act_context_scale
                        x_new.append(u)
                    clean_x = x_new

                clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
                seq_lens_clean = torch.tensor(
                    [u.size(1) for u in clean_x], dtype=torch.long, device=device
                )
                assert seq_lens_clean.max() <= seq_len
                max_len_clean = seq_lens_clean[0].item()
                clean_x = torch.cat([
                    torch.cat(
                        [u, u.new_zeros(1, max_len_clean - u.size(1), u.size(2))], dim=1
                    )
                    for u in clean_x
                ])

                if clean_x.shape[1] != x.shape[1]:
                    _dbg_print(
                        "teacher_forcing.clean_noisy_len_mismatch",
                        clean_x=clean_x,
                        x=x,
                        seq_lens_clean=seq_lens_clean,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                    )
                    raise RuntimeError(
                        f"clean_x token len {clean_x.shape[1]} != noisy x token len {x.shape[1]}"
                    )

                if aug_t is None:
                    aug_t = torch.zeros_like(t)
                e_clean = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).to(x.dtype))
                e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
                if ref_info is not None:
                    ref_tokens = ref_info["tokens"]
                    branch_token_len = x.shape[1]
                    ref_e0 = self._zero_ref_modulation(
                        batch_size=x.shape[0],
                        token_len=N_r,
                        device=device,
                        dtype=x.dtype,
                    )
                    e0 = self._expand_frame_modulation_to_tokens(e0, frame_seqlen)
                    e0_clean = self._expand_frame_modulation_to_tokens(e0_clean, frame_seqlen)
                    x = torch.cat([ref_tokens, clean_x, x], dim=1)
                    e0 = torch.cat([ref_e0, e0_clean, e0], dim=1)
                    noisy_branch_start = ref_token_len + branch_token_len
                else:
                    x = torch.cat([clean_x, x], dim=1)
                    e0 = torch.cat([e0_clean, e0], dim=1)
                    noisy_branch_start = x.shape[1] // 2
            elif ref_info is not None:
                ref_tokens = ref_info["tokens"]
                branch_token_len = x.shape[1]
                ref_e0 = self._zero_ref_modulation(
                    batch_size=x.shape[0],
                    token_len=N_r,
                    device=device,
                    dtype=x.dtype,
                )
                e0 = self._expand_frame_modulation_to_tokens(e0, frame_seqlen)
                x = torch.cat([ref_tokens, x], dim=1)
                e0 = torch.cat([ref_e0, e0], dim=1)
                noisy_branch_start = ref_token_len
            else:
                noisy_branch_start = 0

            for block in self.blocks:
                block.self_attn._is_teacher_forcing = clean_x is not None
                block.self_attn._num_ref_tokens = N_r
                block.self_attn._query_ref_token_len = N_r
                if ref_info is not None:
                    block.self_attn._ref_num_slots = ref_info["num_slots"]
                    block.self_attn._ref_tokens_per_frame = ref_info["tokens_per_slot"]
                    block.self_attn._ref_grid_sizes = ref_info["grid"]
                else:
                    block.self_attn._ref_num_slots = 0
                    block.self_attn._ref_tokens_per_frame = None
                    block.self_attn._ref_grid_sizes = None

            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                block_mask=self.block_mask,
                current_start=current_start,
            )

            def create_custom_forward(module):
                def custom_forward(*inputs, **kw):
                    return module(*inputs, **kw)

                return custom_forward

            for block_idx, block in enumerate(self.blocks):
                try:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, **kwargs, use_reentrant=False,
                        )
                    else:
                        x = block(x, **kwargs)
                except Exception as e_block:
                    if _is_checkpoint_stop_signal(e_block):
                        raise
                    _dbg_print(
                        "model.forward_train.block.failed",
                        error=repr(e_block),
                        block_idx=block_idx,
                        x=x,
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        context=context,
                        block_mask=_dbg_block_mask(self.block_mask),
                        clean_x_is_not_none=(clean_x is not None),
                    )
                    raise

            if ref_info is not None or clean_x is not None:
                x = x[:, noisy_branch_start:]
            elif clean_x is not None:
                x = x[:, x.shape[1] // 2:]

            x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
            x = self.unpatchify(x, grid_sizes)
            return torch.stack(x)
        except Exception as err:
            if _is_checkpoint_stop_signal(err):
                raise
            _dbg_print(
                "model.forward_train.failed",
                error=repr(err),
                t=t,
                seq_len=seq_len,
                y=y,
                clean_x_is_not_none=(clean_x is not None),
                aug_t=aug_t,
                clip_fea=clip_fea,
                act_context=act_context,
            )
            raise

    # ===== 对外 forward：根据是否传 kv_cache 判断 train/inference =====

    # def forward(self, *args, **kwargs):
    #     if kwargs.get('kv_cache', None) is not None:
    #         return self._forward_inference(*args, **kwargs)
    #     else:
    #         return self._forward_train(*args, **kwargs)
    def forward(self, *args, **kwargs):
        # 关键：只要当前在建梯度图，就不要走 kv_cache inference path。
        # 不要依赖 self.training，因为蒸馏/采样训练里经常是 eval() + grad enabled。
        if torch.is_grad_enabled() and kwargs.get("kv_cache", None) is not None:
            for k in ["kv_cache", "crossattn_cache", "current_start", "cache_start"]:
                kwargs.pop(k, None)
            return self._forward_train(*args, **kwargs)

        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)

        return self._forward_train(*args, **kwargs)

    # ===== 其余保持 2.2 一致 =====

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        try:
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
                out.append(u)
            return out
        except Exception as e:
            if _is_checkpoint_stop_signal(e):
                raise
            _dbg_print(
                "unpatchify.failed",
                error=repr(e),
                x=x,
                grid_sizes=grid_sizes,
                out_dim=c,
                patch_size=self.patch_size,
            )
            raise

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)
