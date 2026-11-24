"""Wan video backbone shared by the teacher's dense attention and input routing."""
import math
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from abot_world.bit.wan.modules.model import SimpleAdapter, MLPProj, WanI2VCrossAttention
from einops import rearrange

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

from abot_world.bit.wan.modules.attention import attention
from abot_world.bit.wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    rope_params,
    sinusoidal_embedding_1d,
    WanSelfAttention,
)
from abot_world.bit.wan.modules.attention import flash_attention
from abot_world.bit.utils.chunkwise_prompt import (
    build_chunkwise_cross_attn_ids,
    resolve_cross_attn_num_frames,
)

KV_Cache_Debug=False

def freqs_at(indices, dim, theta=10000, device='cpu'):
    """Compute RoPE freqs for arbitrary integer indices (including negative).

    Unlike ``rope_params`` which builds a table from ``torch.arange(max_seq_len)``,
    this function accepts an explicit list of integer positions so that negative
    temporal indices (used for ref-image tokens) are computed correctly rather
    than being misinterpreted as Python-style reverse indexing.

    Args:
        indices: list/tensor of integer positions.
        dim: half the head dimension for this RoPE axis.
        theta: base frequency (default 10000).
        device: torch device.

    Returns:
        Complex tensor of shape [len(indices), dim // 2].
    """
    assert dim % 2 == 0
    t = torch.tensor(indices, dtype=torch.float64, device=device)
    freqs = torch.outer(
        t,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=device, dtype=torch.float64).div(dim))
    )
    return torch.polar(torch.ones_like(freqs), freqs)

def rope_apply_with_refimg(x, freqs, num_heads):
    if x.dim() == 3:
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # x is now [b, s, n, d]
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(3)
    return x_out.to(x.dtype)

def _inference_ref_freqs(freqs, num_slots, tokens_per_slot, ref_grid, device):
    """Previous DMD 5B reference RoPE, adapted to this model's (H, W) metadata."""
    patch_h, patch_w = [int(v) for v in ref_grid]
    if patch_h * patch_w != tokens_per_slot:
        raise ValueError("Reference token count does not match its spatial grid")
    freq_dim = freqs.shape[1]
    f_band = freq_dim - 2 * (freq_dim // 3)
    h_band = w_band = freq_dim // 3
    temporal_step = max(int(tokens_per_slot), 256)
    neg_temporal = [-(num_slots - i) * temporal_step for i in range(num_slots)]
    t_freqs = freqs_at(neg_temporal, 2 * f_band, device=device)
    freqs_split = freqs.split([f_band, h_band, w_band], dim=1)
    h_freqs = freqs_split[1][:patch_h].to(device)
    w_freqs = freqs_split[2][:patch_w].to(device)
    return torch.cat([
        t_freqs[:, None, None, :].expand(num_slots, patch_h, patch_w, f_band),
        h_freqs[None, :, None, :].expand(num_slots, patch_h, patch_w, h_band),
        w_freqs[None, None, :, :].expand(num_slots, patch_h, patch_w, w_band),
    ], dim=-1).reshape(num_slots * tokens_per_slot, 1, -1)


# =========================================================
# RoPE helpers
# =========================================================

@torch.amp.autocast('cuda', enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    原始 absolute start_frame 版本：
    推理时使用全局绝对时间偏移。
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]], dim=0)
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def relative_rope_apply(x, grid_sizes, freqs, frame_indices=None):
    """
    Relative RoPE（Helios-style 最小实现版）：
    - 不再用全局 absolute start_frame
    - 直接传入当前窗口内的 frame_indices
    - frame_indices 通常是 [0, 1, ..., visible_frames-1]
      或 query 对应窗口尾部的子区间
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )

        if frame_indices is None:
            t_index = torch.arange(f, device=x.device, dtype=torch.long)
        else:
            t_index = frame_indices[:f].to(device=x.device, dtype=torch.long)

        freqs_i = torch.cat([
            freqs[0][t_index].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]], dim=0)
        output.append(x_i)

    return torch.stack(output).float()


# =========================================================
# Self-Attention
# =========================================================

flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)


class WanBackboneSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 use_relative_rope=True):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_relative_rope = use_relative_rope

        # 只用于 KV cache 推理路径
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 880

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,              # [B, L, C]
        seq_lens,
        grid_sizes,
        freqs,
        block_mask: BlockMask | None = None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        ref_only_prefill: bool = False,
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

        q, k, v = qkv_fn(x)

        # Ref-image tokens form a closed attention component in the teacher:
        # ref queries can only see ref K/V, while video queries can see both
        # ref and video K/V.  Therefore their per-layer K/V can be computed
        # once, before the video denoising loop, without changing the result.
        if kv_cache is not None and ref_only_prefill:
            num_ref = int(kv_cache.get("_num_ref_tokens", 0))
            ref_tpf = kv_cache.get(
                "_ref_tokens_per_frame",
                getattr(self, "_ref_tokens_per_frame", None),
            )
            ref_grid_info = kv_cache.get(
                "_ref_grid_sizes",
                getattr(self, "_ref_grid_sizes", None),
            )
            if num_ref <= 0 or s != num_ref:
                raise ValueError(
                    "ref-only KV prefill expects exactly the reserved ref tokens: "
                    f"input={s}, reserved={num_ref}"
                )
            if not ref_tpf or ref_grid_info is None:
                raise ValueError("ref-only KV prefill is missing ref RoPE metadata")

            rH, rW = ref_grid_info
            num_ref_frames = num_ref // int(ref_tpf)
            if num_ref_frames * int(ref_tpf) != num_ref:
                raise ValueError(
                    f"num_ref_tokens={num_ref} is not divisible by "
                    f"ref_tokens_per_frame={ref_tpf}"
                )
            freq_dim = freqs.shape[1]
            f_band = freq_dim - 2 * (freq_dim // 3)
            h_band = freq_dim // 3
            w_band = freq_dim // 3
            temporal_step = max(int(ref_tpf), 256)
            neg_temporal = [
                -(num_ref_frames - i) * temporal_step
                for i in range(num_ref_frames)
            ]
            t_freqs = freqs_at(
                neg_temporal, 2 * f_band, device=q.device
            )
            freqs_split = freqs.split([f_band, h_band, w_band], dim=1)
            h_freqs = freqs_split[1][:rH]
            w_freqs = freqs_split[2][:rW]
            t_part = t_freqs.view(
                num_ref_frames, 1, 1, 1, -1
            ).expand(num_ref_frames, 1, rH, rW, -1)
            h_part = h_freqs.view(
                1, 1, rH, 1, -1
            ).expand(num_ref_frames, 1, rH, rW, -1)
            w_part = w_freqs.view(
                1, 1, 1, rW, -1
            ).expand(num_ref_frames, 1, rH, rW, -1)
            ref_freqs = torch.cat(
                [t_part, h_part, w_part], dim=-1
            ).reshape(num_ref, 1, -1).to(q.device)
            roped_q = rope_apply_with_refimg(q, ref_freqs, self.num_heads)
            roped_k = rope_apply_with_refimg(k, ref_freqs, self.num_heads)

            if self.use_relative_rope:
                if "k_raw" not in kv_cache:
                    kv_cache["k_raw"] = torch.empty_like(kv_cache["k"])
                kv_cache["k_raw"][:, :num_ref] = k
            else:
                kv_cache["k"][:, :num_ref] = roped_k
            kv_cache["v"][:, :num_ref] = v
            kv_cache["local_end_index"].fill_(num_ref)
            kv_cache["global_end_index"].zero_()
            kv_cache["_ref_prefilled"] = True

            x = attention(roped_q, roped_k, v).flatten(2)
            return self.o(x)

        # -------------------------------------------------
        # Training path
        # -------------------------------------------------
        if kv_cache is None:
            num_ref = getattr(self, '_num_ref_tokens', 0)
            is_tf = getattr(self, '_is_teacher_forcing', False) or (s == seq_lens[0].item() * 2)

            if is_tf and num_ref > 0:
                # 3-chunk: [ref | clean | noisy]
                ref_len = num_ref
                half_len = (s - ref_len) // 2

                # Build ref freqs with negative temporal indices
                ref_tpf = getattr(self, '_ref_tokens_per_frame', None)
                if ref_tpf is None or ref_tpf == 0:
                    ref_tpf = grid_sizes[0, 1].item() * grid_sizes[0, 2].item()
                K_frames = ref_len // ref_tpf
                assert ref_len == K_frames * ref_tpf, "ref_len must be divisible by ref_tokens_per_frame"

                # Get ref spatial dims: prefer explicit _ref_grid_sizes, else infer from ref_tpf
                ref_grid_info = getattr(self, '_ref_grid_sizes', None)
                rH, rW = ref_grid_info
                rT = 1  # each ref slot is a single "frame"

                # Temporal: negative indices via freqs_at (match teacher model)
                # freqs is complex tensor, dim=1 is head_dim // 2
                freq_dim = freqs.shape[1]
                f_band = freq_dim - 2 * (freq_dim // 3)
                h_band = freq_dim // 3
                w_band = freq_dim // 3
                tokens_per_slot = ref_tpf
                temporal_step = max(tokens_per_slot, 256)
                neg_temporal = [-(K_frames - i) * temporal_step for i in range(K_frames)]
                t_freqs = freqs_at(neg_temporal, 2 * f_band, device=q.device)

                # Spatial: use precomputed freqs table (split from passed-in freqs)
                freqs_split = freqs.split([f_band, h_band, w_band], dim=1)
                h_freqs = freqs_split[1][:rH]
                w_freqs = freqs_split[2][:rW]

                t_part = t_freqs.view(K_frames, 1, 1, 1, -1).expand(K_frames, rT, rH, rW, -1)
                h_part = h_freqs.view(1, 1, rH, 1, -1).expand(K_frames, rT, rH, rW, -1)
                w_part = w_freqs.view(1, 1, 1, rW, -1).expand(K_frames, rT, rH, rW, -1)
                ref_freqs = torch.cat([t_part, h_part, w_part], dim=-1).reshape(ref_len, 1, -1).to(q.device)

                # Video freqs: standard construction for clean + noisy (2 * num_frames)
                vf = grid_sizes[0, 0].item()
                vh = grid_sizes[0, 1].item()
                vw = grid_sizes[0, 2].item()
                video_freqs = torch.cat([
                    freqs_split[0][:vf].view(vf, 1, 1, -1).expand(vf, vh, vw, -1),
                    freqs_split[1][:vh].view(1, vh, 1, -1).expand(vf, vh, vw, -1),
                    freqs_split[2][:vw].view(1, 1, vw, -1).expand(vf, vh, vw, -1),
                ], dim=-1).reshape(vf * vh * vw, 1, -1)
                # print(f"ref_freqs: {ref_freqs.shape} video_freqs: {video_freqs.shape} q.shape: {q.shape} k.shape: {k.shape} ref_len: {ref_len} half_len: {half_len}")

                # Full freqs: [ref | video (used for both clean and noisy)]
                # For the full sequence [ref | clean | noisy], clean and noisy
                # share the same video freqs (same spatial layout, same temporal order).
                # Apply RoPE per chunk using pre-built per-token freqs
                roped_query = torch.cat([
                    rope_apply_with_refimg(q[:, :ref_len], ref_freqs, self.num_heads),
                    rope_apply_with_refimg(q[:, ref_len:ref_len + half_len], video_freqs, self.num_heads),
                    rope_apply_with_refimg(q[:, ref_len + half_len:], video_freqs, self.num_heads),
                ], dim=1)
                roped_key = torch.cat([
                    rope_apply_with_refimg(k[:, :ref_len], ref_freqs, self.num_heads),
                    rope_apply_with_refimg(k[:, ref_len:ref_len + half_len], video_freqs, self.num_heads),
                    rope_apply_with_refimg(k[:, ref_len + half_len:], video_freqs, self.num_heads),
                ], dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                if padded_length > 0:
                    padded_roped_query = torch.cat(
                        [roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)], dim=1)
                    padded_roped_key = torch.cat(
                        [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)], dim=1)
                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1), key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1), block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    x = flex_attention(
                        query=roped_query.transpose(2, 1), key=roped_key.transpose(2, 1),
                        value=v.transpose(2, 1), block_mask=block_mask
                    ).transpose(2, 1)

            elif is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []

                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                if padded_length > 0:
                    padded_roped_query = torch.cat(
                        [
                            roped_query,
                            torch.zeros(
                                [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype
                            )
                        ],
                        dim=1
                    )

                    padded_roped_key = torch.cat(
                        [
                            roped_key,
                            torch.zeros(
                                [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                device=k.device, dtype=v.dtype
                            )
                        ],
                        dim=1
                    )

                    padded_v = torch.cat(
                        [
                            v,
                            torch.zeros(
                                [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype
                            )
                        ],
                        dim=1
                    )

                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    x = flex_attention(
                        query=roped_query.transpose(2, 1),
                        key=roped_key.transpose(2, 1),
                        value=v.transpose(2, 1),
                        block_mask=block_mask
                    ).transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                if padded_length > 0:
                    padded_roped_query = torch.cat(
                        [
                            roped_query,
                            torch.zeros(
                                [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype
                            )
                        ],
                        dim=1
                    )

                    padded_roped_key = torch.cat(
                        [
                            roped_key,
                            torch.zeros(
                                [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                device=k.device, dtype=v.dtype
                            )
                        ],
                        dim=1
                    )

                    padded_v = torch.cat(
                        [
                            v,
                            torch.zeros(
                                [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype
                            )
                        ],
                        dim=1
                    )

                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    x = flex_attention(
                        query=roped_query.transpose(2, 1),
                        key=roped_key.transpose(2, 1),
                        value=v.transpose(2, 1),
                        block_mask=block_mask
                    ).transpose(2, 1)

        # -------------------------------------------------
        # Inference path
        # -------------------------------------------------
        else:
            frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
            num_new_tokens = q.shape[1]
            num_new_frames = int(grid_sizes[0][0].item())

            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen

            # =============================================
            # Relative RoPE path (default)
            # =============================================
            if self.use_relative_rope:
                # Port of 5B_inference's separate reference/video relative RoPE.
                # This cache protocol keeps global_end video-only, local_end includes refs.
                num_ref = int(kv_cache.get("_num_ref_tokens", 0))
                ref_tpf = getattr(self, "_ref_tokens_per_frame", None)
                ref_grid_info = getattr(self, "_ref_grid_sizes", None)
                video_tokens = int(grid_sizes[0][0].item()) * frame_seqlen
                query_ref_tokens = q.shape[1] - video_tokens
                if query_ref_tokens not in (0, num_ref):
                    raise ValueError("Relative KV reference prefix does not match input geometry")
                video_current_end = current_start + video_tokens
                effective_sink = num_ref + sink_tokens

                if "k_raw" not in kv_cache:
                    kv_cache["k_raw"] = torch.empty_like(kv_cache["k"])
                cache_size = kv_cache["k_raw"].shape[1]
                previous_local_end = int(kv_cache["local_end_index"].item())
                previous_global_end = int(kv_cache["global_end_index"].item())
                evicted = 0
                if (self.local_attn_size != -1 and video_current_end > previous_global_end
                        and video_tokens + previous_local_end > cache_size):
                    evicted = video_tokens + previous_local_end - cache_size
                    rolled = previous_local_end - evicted - effective_sink
                    if rolled < 0:
                        raise RuntimeError("Relative KV block exceeds cache video capacity")
                    for name in ("k_raw", "v"):
                        kv_cache[name][:, effective_sink:effective_sink + rolled] = kv_cache[name][
                            :, effective_sink + evicted:effective_sink + evicted + rolled
                        ].clone()
                local_end_index = previous_local_end + video_current_end - previous_global_end - evicted
                video_start = local_end_index - video_tokens
                if video_start < num_ref or local_end_index > cache_size:
                    raise RuntimeError("Relative KV video write overlaps references or exceeds capacity")
                if query_ref_tokens:
                    kv_cache["k_raw"][:, :num_ref] = k[:, :query_ref_tokens].detach()
                    kv_cache["v"][:, :num_ref] = v[:, :query_ref_tokens]
                kv_cache["k_raw"][:, video_start:local_end_index] = k[:, query_ref_tokens:].detach()
                kv_cache["v"][:, video_start:local_end_index] = v[:, query_ref_tokens:]

                max_video_tokens = (local_end_index - effective_sink if self.local_attn_size == -1
                                    else self.local_attn_size * frame_seqlen)
                recent_start = max(effective_sink, local_end_index - max_video_tokens)
                recent_start += (local_end_index - recent_start) % frame_seqlen
                video_key_parts, video_value_parts = [], []
                # Only initialized video sinks are visible during the first block.
                sink_end = min(effective_sink, local_end_index)
                if sink_end > num_ref:
                    video_key_parts.append(kv_cache["k_raw"][:, num_ref:sink_end])
                    video_value_parts.append(kv_cache["v"][:, num_ref:sink_end])
                if local_end_index > recent_start:
                    video_key_parts.append(kv_cache["k_raw"][:, recent_start:local_end_index])
                    video_value_parts.append(kv_cache["v"][:, recent_start:local_end_index])
                visible_video_raw = torch.cat(video_key_parts, dim=1)
                visible_video_v = torch.cat(video_value_parts, dim=1)
                visible_frames = visible_video_raw.shape[1] // frame_seqlen
                visible_grid = grid_sizes.clone()
                visible_grid[:, 0] = visible_frames
                key_positions = torch.arange(visible_frames, device=q.device, dtype=torch.long)
                query_positions = key_positions[-int(grid_sizes[0][0].item()):]
                roped_query = relative_rope_apply(
                    q[:, query_ref_tokens:], grid_sizes, freqs, frame_indices=query_positions
                ).type_as(v)
                roped_key = relative_rope_apply(
                    visible_video_raw, visible_grid, freqs, frame_indices=key_positions
                ).type_as(v)

                if num_ref:
                    if not ref_tpf or ref_grid_info is None or num_ref % ref_tpf:
                        raise RuntimeError("Reference RoPE metadata is incomplete")
                    ref_freqs = _inference_ref_freqs(
                        freqs, num_ref // ref_tpf, ref_tpf, ref_grid_info, q.device
                    )
                    ref_key = rope_apply_with_refimg(
                        kv_cache["k_raw"][:, :num_ref], ref_freqs, self.num_heads
                    ).type_as(v)
                    roped_key = torch.cat([ref_key, roped_key], dim=1)
                    visible_video_v = torch.cat([kv_cache["v"][:, :num_ref], visible_video_v], dim=1)
                    if query_ref_tokens:
                        ref_query = rope_apply_with_refimg(
                            q[:, :query_ref_tokens], ref_freqs, self.num_heads
                        ).type_as(v)
                        roped_query = torch.cat([ref_query, roped_query], dim=1)
                x = attention(roped_query, roped_key, visible_video_v)
                kv_cache["global_end_index"].fill_(video_current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            # =============================================
            # Original absolute RoPE path
            # =============================================
            else:
                num_ref = getattr(self, '_num_ref_tokens', 0)
                ref_tpf = getattr(self, '_ref_tokens_per_frame', None)
                ref_grid_info = getattr(self, '_ref_grid_sizes', None)

                current_start_frame = current_start // frame_seqlen

                # Split ref and video tokens BEFORE applying RoPE
                video_frames_in_input = int(grid_sizes[0][0].item())
                video_tokens_in_input = video_frames_in_input * frame_seqlen
                ref_in_input = max(0, num_new_tokens - video_tokens_in_input)

                if ref_in_input > 0:
                    # Has ref tokens: apply RoPE separately
                    ref_q = q[:, :ref_in_input]
                    video_q = q[:, ref_in_input:]
                    ref_k = k[:, :ref_in_input]
                    video_k = k[:, ref_in_input:]

                    # Apply video RoPE to video tokens only
                    video_roped_q = causal_rope_apply(
                        video_q, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)
                    video_roped_k = causal_rope_apply(
                        video_k, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)

                    # Apply ref RoPE with negative temporal indices
                    if ref_tpf is not None and ref_grid_info is not None:
                        rH, rW = ref_grid_info
                        K_frames = num_ref // ref_tpf
                        freq_dim = freqs.shape[1]
                        f_band = freq_dim - 2 * (freq_dim // 3)
                        h_band = freq_dim // 3
                        w_band = freq_dim // 3

                        temporal_step = max(ref_tpf, 256)
                        neg_temporal = [-(K_frames - i) * temporal_step for i in range(K_frames)]
                        t_freqs = freqs_at(neg_temporal, 2 * f_band, device=q.device)

                        freqs_split = freqs.split([f_band, h_band, w_band], dim=1)
                        h_freqs = freqs_split[1][:rH]
                        w_freqs = freqs_split[2][:rW]

                        rT = 1
                        t_part = t_freqs.view(K_frames, 1, 1, 1, -1).expand(K_frames, rT, rH, rW, -1)
                        h_part = h_freqs.view(1, 1, rH, 1, -1).expand(K_frames, rT, rH, rW, -1)
                        w_part = w_freqs.view(1, 1, 1, rW, -1).expand(K_frames, rT, rH, rW, -1)
                        ref_freqs = torch.cat([t_part, h_part, w_part], dim=-1).reshape(num_ref, -1).to(q.device)

                        ref_grid = torch.tensor([[K_frames, rH, rW]], device=q.device, dtype=grid_sizes.dtype)
                        ref_roped_q = rope_apply(ref_q, ref_grid, ref_freqs).type_as(v)
                        ref_roped_k = rope_apply(ref_k, ref_grid, ref_freqs).type_as(v)
                    else:
                        # No ref RoPE config, just use identity
                        ref_roped_q = ref_q
                        ref_roped_k = ref_k

                    # Concatenate ref and video
                    roped_query = torch.cat([ref_roped_q, video_roped_q], dim=1)
                    roped_key = torch.cat([ref_roped_k, video_roped_k], dim=1)
                else:
                    # No ref tokens: apply video RoPE to all tokens
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)
                    roped_key = causal_rope_apply(
                        k, grid_sizes, freqs, start_frame=current_start_frame
                    ).type_as(v)

                kv_cache_size = kv_cache["k"].shape[1]
                num_ref = kv_cache.get("_num_ref_tokens", 0)
                # Ref tokens are permanent sink: protect from eviction
                effective_sink = sink_tokens + num_ref
                # Calculate how many ref tokens are in THIS input
                video_frames_in_input = int(grid_sizes[0][0].item())
                video_tokens_in_input = video_frames_in_input * frame_seqlen #当前输入中所有 video tokens 的数量
                ref_in_input = max(0, num_new_tokens - video_tokens_in_input)
                # Use video_current_end for consistent indexing
                video_current_end = current_start + video_tokens_in_input

                # Cache layout: [0, num_ref) = ref tokens, [sink_offset, effective_sink) = sink, [effective_sink, ...) = video
                # local_end_index tracks the absolute cache position of the END of stored video tokens
                # It starts at effective_sink (after ref+sink) and grows with each video chunk

                # [KV-Cache-Debug] Print step-level summary (absolute RoPE path)
                if KV_Cache_Debug:
                    print(f"[KV-Cache-Debug] current_start={current_start}, num_new_tokens={num_new_tokens}, "
                      f"num_ref={num_ref}, ref_in_input={ref_in_input}, "
                      f"video_tokens_in_input={video_tokens_in_input}, video_current_end={video_current_end}, "
                      f"global_end_index={kv_cache['global_end_index'].item()}, "
                      f"local_end_index={kv_cache['local_end_index'].item()}, kv_cache_size={kv_cache_size}")

                if self.local_attn_size != -1 and (video_current_end > kv_cache["global_end_index"].item()) and (
                    video_tokens_in_input + kv_cache["local_end_index"].item() > kv_cache_size
                ):
                    # Eviction path: need to roll video tokens left to free space
                    # available_video_space = kv_cache_size - effective_sink
                    num_evicted_tokens = video_tokens_in_input + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - effective_sink
                    # Roll only video tokens (ref tokens at [0, num_ref) stay)
                    # [KV-Cache-Debug] Print eviction details
                    if KV_Cache_Debug:
                        print(f"[KV-Cache-Debug] EVICTION: num_evicted_tokens={num_evicted_tokens}, "
                          f"num_rolled_tokens={num_rolled_tokens}, "
                          f"rolling video tokens [{effective_sink}:{effective_sink + num_evicted_tokens}) -> [{effective_sink}:{effective_sink + num_rolled_tokens})")
                    kv_cache["k"][:, effective_sink:effective_sink + num_rolled_tokens] = \
                        kv_cache["k"][:, effective_sink + num_evicted_tokens:effective_sink + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, effective_sink:effective_sink + num_rolled_tokens] = \
                        kv_cache["v"][:, effective_sink + num_evicted_tokens:effective_sink + num_evicted_tokens + num_rolled_tokens].clone()

                    # local_end_index: absolute end position in cache (ref+sink+video_all)
                    # = prev_local_end + new_video_tokens_global_offset - evicted_tokens
                    local_end_index = kv_cache["local_end_index"].item() + video_current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    video_start = local_end_index - video_tokens_in_input
                    video_end = local_end_index
                    # Write ref tokens to [0, num_ref) (idempotent) and video tokens to [video_start, video_end)
                    if ref_in_input > 0:
                        # [KV-Cache-Debug] Print write positions (eviction path)
                        if KV_Cache_Debug:
                            print(f"[KV-Cache-Debug] WRITE (eviction path): [REF] cache[0:{num_ref}) = input[0:{ref_in_input}), "
                              f"[VIDEO] cache[{video_start}:{video_end}) = input[{ref_in_input}:{num_new_tokens})") 
                        kv_cache["k"][:, :num_ref] = roped_key[:, :ref_in_input]
                        kv_cache["v"][:, :num_ref] = v[:, :ref_in_input]
                        kv_cache["k"][:, video_start:video_end] = roped_key[:, ref_in_input:]
                        kv_cache["v"][:, video_start:video_end] = v[:, ref_in_input:]
                    else:
                        # [KV-Cache-Debug] Print write positions (eviction path, no ref in input)
                        if KV_Cache_Debug:
                            print(f"[KV-Cache-Debug] WRITE (eviction path): [VIDEO] cache[{video_start}:{video_end}) = input[0:{num_new_tokens})")
                        kv_cache["k"][:, video_start:video_end] = roped_key
                        kv_cache["v"][:, video_start:video_end] = v
                else:
                    # No eviction: simply append video tokens after existing content
                    local_end_index = kv_cache["local_end_index"].item() + video_current_end - kv_cache["global_end_index"].item()
                    video_start = local_end_index - video_tokens_in_input
                    video_end = local_end_index
                    # Write ref tokens to [0, num_ref) (idempotent) and video tokens to [video_start, video_end)
                    if ref_in_input > 0:
                        # [KV-Cache-Debug] Print write positions (no-eviction path)
                        if KV_Cache_Debug:
                            print(f"[KV-Cache-Debug] WRITE (no-eviction path): [REF] cache[0:{num_ref}) = input[0:{ref_in_input}), "
                              f"[VIDEO] cache[{video_start}:{video_end}) = input[{ref_in_input}:{num_new_tokens})")
                        kv_cache["k"][:, :num_ref] = roped_key[:, :ref_in_input]
                        kv_cache["v"][:, :num_ref] = v[:, :ref_in_input]
                        kv_cache["k"][:, video_start:video_end] = roped_key[:, ref_in_input:]
                        kv_cache["v"][:, video_start:video_end] = v[:, ref_in_input:]
                    else:
                        # [KV-Cache-Debug] Print write positions (no-eviction path, no ref in input)
                        if KV_Cache_Debug:
                            print(f"[KV-Cache-Debug] WRITE (no-eviction path): [VIDEO] cache[{video_start}:{video_end}) = input[0:{num_new_tokens})")
                        kv_cache["k"][:, video_start:video_end] = roped_key
                        kv_cache["v"][:, video_start:video_end] = v

                # Ensure ref tokens are always in the attention window
                attn_start = max(0, local_end_index - self.max_attention_size)
                if num_ref > 0 and attn_start > 0:
                    attn_start = 0
                # [KV-Cache-Debug] Print attention window
                if KV_Cache_Debug:
                    print(f"[KV-Cache-Debug] ATTENTION: attn_start={attn_start}, local_end_index={local_end_index}, "
                      f"window_size={local_end_index - attn_start}, "
                      f"ref_in_window={'YES (attn_start=0)' if num_ref > 0 and attn_start == 0 else 'NO'}, "
                      f"max_attention_size={self.max_attention_size}")
                window = kv_cache.get("_tfroute_video_window")
                if window is not None:
                    # TF clean and noisy branches have distinct video windows;
                    # the permanent ref/sink prefix must not expand that window.
                    video_begin = max(effective_sink, local_end_index - int(window) * frame_seqlen)
                    if effective_sink and video_begin > effective_sink:
                        visible_k = torch.cat([kv_cache["k"][:, :effective_sink], kv_cache["k"][:, video_begin:local_end_index]], dim=1)
                        visible_v = torch.cat([kv_cache["v"][:, :effective_sink], kv_cache["v"][:, video_begin:local_end_index]], dim=1)
                    else:
                        begin = 0 if effective_sink else video_begin
                        visible_k = kv_cache["k"][:, begin:local_end_index]
                        visible_v = kv_cache["v"][:, begin:local_end_index]
                else:
                    visible_k = kv_cache["k"][:, attn_start:local_end_index]
                    visible_v = kv_cache["v"][:, attn_start:local_end_index]
                x = attention(roped_query, visible_k, visible_v)
                kv_cache["global_end_index"].fill_(video_current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x


# =========================================================
# Cross-Attention
# =========================================================

class CausalWanCrossAttention(WanSelfAttention):

    def _chunkwise_forward(
        self,
        x,
        context,
        chunk_ids,
        crossattn_cache=None,
    ):
        """Route each query token to one independently encoded text chunk."""

        if context.ndim != 4:
            raise ValueError(
                f"chunk-wise context must be [B,C,L,D], got {tuple(context.shape)}"
            )
        if chunk_ids is None:
            raise ValueError("chunk-wise context requires cross_attn_chunk_ids")
        if chunk_ids.ndim != 2 or chunk_ids.shape[:2] != x.shape[:2]:
            raise ValueError(
                "cross_attn_chunk_ids must match query shape [B,S], got "
                f"{tuple(chunk_ids.shape)} for x={tuple(x.shape)}"
            )
        if context.shape[0] != x.shape[0]:
            raise ValueError(
                f"context batch {context.shape[0]} does not match query batch {x.shape[0]}"
            )

        groups = []
        for batch_id in range(x.shape[0]):
            for chunk_id in range(context.shape[1]):
                indices = torch.nonzero(
                    chunk_ids[batch_id] == chunk_id, as_tuple=False
                ).flatten()
                if indices.numel() > 0:
                    groups.append((batch_id, chunk_id, indices))
        if not groups:
            return torch.zeros_like(x)

        b, n, d = x.size(0), self.num_heads, self.head_dim
        max_queries = max(indices.numel() for _, _, indices in groups)
        grouped_x = x.new_zeros((len(groups), max_queries, x.shape[-1]))
        flat_indices = []
        valid_lengths = []
        for group_id, (batch_id, _, indices) in enumerate(groups):
            grouped_x[group_id, :indices.numel()] = x[batch_id].index_select(
                0, indices
            )
            flat_indices.append(indices + batch_id * x.shape[1])
            valid_lengths.append(indices.numel())

        q = self.norm_q(self.q(grouped_x)).view(
            len(groups), max_queries, n, d
        )
        if crossattn_cache is not None and crossattn_cache.get("is_init", False):
            all_k = crossattn_cache["k"]
            all_v = crossattn_cache["v"]
        else:
            all_k = self.norm_k(self.k(context)).view(
                context.shape[0], context.shape[1], context.shape[2], n, d
            )
            all_v = self.v(context).view(
                context.shape[0], context.shape[1], context.shape[2], n, d
            )
            if crossattn_cache is not None:
                crossattn_cache["is_init"] = True
                crossattn_cache["k"] = all_k
                crossattn_cache["v"] = all_v

        grouped_k = torch.stack(
            [all_k[batch_id, chunk_id] for batch_id, chunk_id, _ in groups], dim=0
        )
        grouped_v = torch.stack(
            [all_v[batch_id, chunk_id] for batch_id, chunk_id, _ in groups], dim=0
        )
        attended = flash_attention(q, grouped_k, grouped_v)
        attended = self.o(attended.flatten(2))

        valid_outputs = [
            attended[group_id, :valid_length]
            for group_id, valid_length in enumerate(valid_lengths)
        ]
        output = attended.new_zeros((b * x.shape[1], x.shape[2]))
        output = output.index_copy(
            0, torch.cat(flat_indices), torch.cat(valid_outputs, dim=0)
        )
        return output.view_as(x)

    def forward(
        self,
        x,
        context,
        context_lens,
        crossattn_cache=None,
        chunk_ids=None,
    ):
        if context.ndim == 4:
            return self._chunkwise_forward(
                x,
                context,
                chunk_ids,
                crossattn_cache=crossattn_cache,
            )
        if chunk_ids is not None:
            raise ValueError(
                "cross_attn_chunk_ids were provided for a non-chunkwise context"
            )
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

        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanI2VCrossAttention(WanI2VCrossAttention):

    def forward(
        self,
        x,
        context,
        context_lens,
        crossattn_cache=None,
        chunk_ids=None,
    ):
        if chunk_ids is not None or context.ndim == 4:
            raise ValueError(
                "chunk-wise prompt routing is not implemented for i2v cross-attention"
            )
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

        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)

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
                 use_relative_rope=True):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanBackboneSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
            use_relative_rope=use_relative_rope
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

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
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
        num_ref_tokens: int = 0,
        num_ref_frames: int = 0,
        ref_tokens_per_frame: int | None = None,
        cross_attn_chunk_ids: torch.Tensor | None = None,
        ref_only_prefill: bool = False,
    ):
        # e is frame-level: [B, total_frames, 6, dim]
        # total_frames = num_ref_frames + num_video_frames (+ num_video_frames if TF)
        # x layout: [ref_tokens | video_tokens] or [ref_tokens | clean_tokens | noisy_tokens]
        # ref and video may have different tokens_per_frame, so we do AdaLN separately.

        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # Split e into ref and video parts
        if num_ref_tokens > 0 and num_ref_frames > 0:
            e_ref = [ei[:, :num_ref_frames] for ei in e]
            e_vid = [ei[:, num_ref_frames:] for ei in e]
        else:
            e_ref = None
            e_vid = e

        # Helper: apply AdaLN shift+scale with per-frame broadcast
        def apply_adaln(normed, e_shift, e_scale, num_frames, tokens_per_frame):
            return (normed.unflatten(dim=1, sizes=(num_frames, tokens_per_frame)) * (1 + e_scale) + e_shift).flatten(1, 2)

        def apply_gate(y, e_gate, num_frames, tokens_per_frame):
            return (y.unflatten(dim=1, sizes=(num_frames, tokens_per_frame)) * e_gate).flatten(1, 2)

        # Compute per-segment tokens_per_frame
        if e_ref is not None:
            ref_tpf = ref_tokens_per_frame if ref_tokens_per_frame is not None else 1
            num_vid_e_frames = e_vid[0].shape[1]
            vid_tokens = x.shape[1] - num_ref_tokens
            vid_tpf = vid_tokens // num_vid_e_frames if num_vid_e_frames > 0 else 1
        else:
            num_frames = e[0].shape[1]
            frame_seqlen = x.shape[1] // num_frames

        # --- Self-Attention with AdaLN ---
        normed_x = self.norm1(x)

        if e_ref is not None:
            mod_x = torch.cat([
                apply_adaln(normed_x[:, :num_ref_tokens], e_ref[0], e_ref[1], num_ref_frames, ref_tpf),
                apply_adaln(normed_x[:, num_ref_tokens:], e_vid[0], e_vid[1], num_vid_e_frames, vid_tpf),
            ], dim=1)
        else:
            mod_x = apply_adaln(normed_x, e[0], e[1], num_frames, frame_seqlen)

        y = self.self_attn(
            mod_x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=cache_start,
            ref_only_prefill=ref_only_prefill,
        )

        # Gate after self-attention (e[2])
        if e_ref is not None:
            x = x + torch.cat([
                apply_gate(y[:, :num_ref_tokens], e_ref[2], num_ref_frames, ref_tpf),
                apply_gate(y[:, num_ref_tokens:], e_vid[2], num_vid_e_frames, vid_tpf),
            ], dim=1)
        else:
            x = x + apply_gate(y, e[2], num_frames, frame_seqlen)

        # --- Cross-Attention + FFN ---
        def cross_attn_ffn(x, context, context_lens, e, e_ref=None, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                crossattn_cache=crossattn_cache,
                chunk_ids=cross_attn_chunk_ids,
            )

            normed_x = self.norm2(x)
            if e_ref is not None:
                ffn_input = torch.cat([
                    apply_adaln(normed_x[:, :num_ref_tokens], e_ref[3], e_ref[4], num_ref_frames, ref_tpf),
                    apply_adaln(normed_x[:, num_ref_tokens:], e_vid[3], e_vid[4], num_vid_e_frames, vid_tpf),
                ], dim=1)
            else:
                ffn_input = apply_adaln(normed_x, e[3], e[4], num_frames, frame_seqlen)

            y = self.ffn(ffn_input)

            # Gate after FFN (e[5])
            if e_ref is not None:
                x = x + torch.cat([
                    apply_gate(y[:, :num_ref_tokens], e_ref[5], num_ref_frames, ref_tpf),
                    apply_gate(y[:, num_ref_tokens:], e_vid[5], num_vid_e_frames, vid_tpf),
                ], dim=1)
            else:
                x = x + apply_gate(y, e[5], num_frames, frame_seqlen)

            return x

        x = cross_attn_ffn(x, context, context_lens, e, e_ref, crossattn_cache)
        return x


# =========================================================
# Head
# =========================================================

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

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        )
        return x


# =========================================================
# Model
# =========================================================

class WanVideoBackbone(ModelMixin, ConfigMixin):
    """
    基于 Wan 2.2 结构，加入：
    - causal / blockwise / local attention
    - flex_attention + BlockMask 训练
    - KV cache 推理
    - teacher forcing
    - Relative RoPE 开关（默认关闭）
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
                 downscale_factor_control_adapter=8,
                 independent_first_frame=False):
        super().__init__()
        print(f"Initializing WanVideoBackbone with model_type={model_type}, use_relative_rope={use_relative_rope}")
        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
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
        self.use_relative_rope = use_relative_rope

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

        self.act_control_adapter = None

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        if model_type == 'ti2v':
            self.act_control_adapter = SimpleAdapter(
                act_control_in_dim,
                self.dim,
                kernel_size=self.patch_size[1:],
                stride=self.patch_size[1:],
                downscale_factor=downscale_factor_control_adapter
            )
            self._freeze_module(self.act_control_adapter)

        # 3D RoPE freqs，长度上限 1024
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
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame

    @staticmethod
    def _freeze_module(module):
        if module is None:
            return
        module.requires_grad_(False)
        module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.act_control_adapter is not None:
            self.act_control_adapter.eval()
        return self

    def _embed_text_context(self, context):
        """Pad ordinary or chunk-wise T5 embeddings on their token axis."""

        if not isinstance(context, torch.Tensor) or context.ndim not in (3, 4):
            raise ValueError(
                "text context must be [B,L,D] or [B,C,L,D], got "
                f"{type(context).__name__} "
                f"{getattr(context, 'shape', None)}"
            )
        token_length = context.shape[-2]
        if token_length > self.text_len:
            raise ValueError(
                f"text context length {token_length} exceeds model limit {self.text_len}"
            )
        if token_length < self.text_len:
            pad_shape = (*context.shape[:-2], self.text_len - token_length, context.shape[-1])
            context = torch.cat([context, context.new_zeros(pad_shape)], dim=-2)
        return self.text_embedding(context)

    @staticmethod
    def _append_clip_context(context, context_clip):
        if context.ndim == 4:
            context_clip = context_clip.unsqueeze(1).expand(
                -1, context.shape[1], -1, -1
            )
            return torch.cat([context_clip, context], dim=2)
        return torch.cat([context_clip, context], dim=1)

    @staticmethod
    def _build_chunk_routes(
        context,
        frame_ranges,
        raw_num_frames,
        *,
        latent_frames,
        spatial_tokens_per_frame,
        prefix_tokens=0,
        video_copies=1,
        latent_frame_offset=0,
        total_latent_frames=None,
    ):
        if context.ndim != 4:
            if frame_ranges is not None:
                raise ValueError(
                    "cross_attn_frame_ranges require [B,C,L,D] prompt embeddings"
                )
            return None
        if frame_ranges is None or raw_num_frames is None:
            raise ValueError(
                "chunk-wise prompt embeddings require cross_attn_frame_ranges "
                "and cross_attn_num_frames"
            )
        raw_num_frames = resolve_cross_attn_num_frames(
            raw_num_frames, batch_size=context.shape[0]
        )
        frame_ranges = torch.as_tensor(
            frame_ranges, dtype=torch.long, device=context.device
        )
        if frame_ranges.ndim == 2:
            frame_ranges = frame_ranges.unsqueeze(0)
        if frame_ranges.shape[0] != context.shape[0]:
            raise ValueError(
                "chunk frame-range batch does not match prompt context batch: "
                f"ranges={tuple(frame_ranges.shape)}, context={tuple(context.shape)}"
            )
        if frame_ranges.shape[1] != context.shape[1]:
            raise ValueError(
                "chunk frame-range count does not match prompt chunk count: "
                f"ranges={frame_ranges.shape[1]}, context={context.shape[1]}"
            )

        video_routes = build_chunkwise_cross_attn_ids(
            frame_ranges,
            num_frames=raw_num_frames,
            latent_frames=latent_frames,
            spatial_tokens_per_frame=spatial_tokens_per_frame,
            latent_frame_offset=latent_frame_offset,
            total_latent_frames=total_latent_frames,
            device=context.device,
        )
        prefix = torch.full(
            (context.shape[0], int(prefix_tokens)),
            -1,
            dtype=torch.long,
            device=context.device,
        )
        return torch.cat([prefix] + [video_routes] * int(video_copies), dim=1)

    def _set_gradient_checkpointing(self, module, value: bool = False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

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
    def _prepare_teacher_forcing_mask_bk(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block=1,
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

        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )

        for start in frame_indices:
            context_ends[start:start + attention_block_size] = (
                start + attention_block_size
            )

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen,
            total_length,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        for block_index, (start, end) in enumerate(
            zip(noisy_image_start_list, noisy_image_end_list)
        ):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = block_index * attention_block_size

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
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen

        context_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        context_starts = torch.zeros(
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

        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )

        for idx, start in enumerate(frame_indices):
            context_ends[start:start + attention_block_size] = (
                start + attention_block_size
            )
            if local_attn_size == -1:
                context_starts[start:start + attention_block_size] = 0
            else:
                # 统一用 latent 帧数为单位，与 blockwise 函数一致
                visible_start = max(0, start + attention_block_size - local_attn_size * frame_seqlen)
                context_starts[start:start + attention_block_size] = visible_start

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen,
            total_length,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        for block_index, (start, end) in enumerate(
            zip(noisy_image_start_list, noisy_image_end_list)
        ):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = block_index * attention_block_size
            if local_attn_size == -1:
                noise_context_starts[start:end] = 0
            else:
                # 统一用 latent 帧数为单位
                visible_start = max(0, block_index * attention_block_size - local_attn_size * frame_seqlen)
                noise_context_starts[start:end] = visible_start

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])

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
            _compile=True,
            device=device,
        )
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask_i2v(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        Teacher forcing mask with independent first frame (i2v).
        Layout: [clean_frame0 | clean_block0 | ... | clean_blockN | noisy_frame0 | noisy_block0 | ... | noisy_blockN]
        
        The first frame is independent (1 frame), remaining frames are grouped
        into blocks of num_frame_per_block.
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        attention_block_size = frame_seqlen * num_frame_per_block

        context_ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )
        context_starts = torch.zeros(
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

        # === Clean part: first frame independent, then blockwise ===
        # First frame: only attend to itself
        context_ends[:frame_seqlen] = frame_seqlen
        context_starts[:frame_seqlen] = 0

        # Remaining clean frames: blockwise from frame_seqlen onwards
        remaining_clean_indices = torch.arange(
            start=frame_seqlen,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )

        for start in remaining_clean_indices:
            context_ends[start:start + attention_block_size] = (
                start + attention_block_size
            )
            if local_attn_size == -1:
                context_starts[start:start + attention_block_size] = 0
            else:
                visible_start = max(0, start + attention_block_size - local_attn_size * frame_seqlen)
                context_starts[start:start + attention_block_size] = visible_start

        # === Noisy part: first frame independent, then blockwise ===
        noisy_offset = num_frames * frame_seqlen

        # Noisy first frame: attend to clean first frame + self
        noisy_first_start = noisy_offset
        noisy_first_end = noisy_offset + frame_seqlen
        noise_noise_starts[noisy_first_start:noisy_first_end] = noisy_first_start
        noise_noise_ends[noisy_first_start:noisy_first_end] = noisy_first_end
        noise_context_starts[noisy_first_start:noisy_first_end] = 0
        noise_context_ends[noisy_first_start:noisy_first_end] = 0  # no clean context before first frame

        # Remaining noisy blocks
        noisy_block_starts = torch.arange(
            noisy_offset + frame_seqlen,
            noisy_offset + num_frames * frame_seqlen,
            step=attention_block_size,
            device=device,
            dtype=torch.long,
        )
        noisy_block_ends = noisy_block_starts + attention_block_size

        for block_index, (start, end) in enumerate(
            zip(noisy_block_starts, noisy_block_ends)
        ):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # This noisy block can attend to clean tokens up to the corresponding clean block
            # block_index=0 corresponds to clean frames [0, frame_seqlen + attention_block_size)
            # i.e., first frame + first block
            clean_visible_end = frame_seqlen + block_index * attention_block_size
            noise_context_ends[start:end] = clean_visible_end
            if local_attn_size == -1:
                noise_context_starts[start:end] = 0
            else:
                visible_start = max(0, clean_visible_end - local_attn_size * frame_seqlen)
                noise_context_starts[start:end] = visible_start

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])

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
            _compile=True,
            device=device,
        )
        return block_mask

    @staticmethod
    def _debug_print_mask_with_ref(
        num_ref_frames: int,
        ref_tokens_per_frame: int,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        device: torch.device | str = "cpu",
    ):
        """Print mask matrix matching _prepare_teacher_forcing_mask_with_ref logic.

        Layout: [ref(K) | clean(F) | noisy(F)]
        Each ref frame is one block of size ref_tokens_per_frame.
        Each video frame is one block of size frame_seqlen.
        """
        if ref_tokens_per_frame is None:
            ref_tokens_per_frame = frame_seqlen
        ref_length = num_ref_frames * ref_tokens_per_frame
        video_length = num_frames * frame_seqlen
        total_length = ref_length + 2 * video_length
        attention_block_size = frame_seqlen * num_frame_per_block
        clean_ends = ref_length + video_length

        # Replicate the same arrays as the constructor
        context_ends = torch.zeros(total_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length, device=device, dtype=torch.long)

        # Ref blocks
        if ref_length > 0:
            ref_block_starts = torch.arange(0, ref_length, step=ref_tokens_per_frame, device=device, dtype=torch.long)
            for start in ref_block_starts:
                end = min(start + ref_tokens_per_frame, ref_length)
                context_ends[start:end] = ref_length
                context_starts[start:end] = 0

        # Clean blocks
        clean_frame_indices = torch.arange(ref_length, ref_length + video_length, step=attention_block_size, device=device, dtype=torch.long)
        for start in clean_frame_indices:
            end = min(start + attention_block_size, ref_length + video_length)
            context_ends[start:end] = end
            if local_attn_size == -1:
                context_starts[start:end] = 0
            else:
                visible_start = max(0, end - local_attn_size * frame_seqlen)
                context_starts[start:end] = visible_start

        # Noisy blocks
        noisy_start = ref_length + video_length
        noisy_frame_indices = torch.arange(noisy_start, total_length, step=attention_block_size, device=device, dtype=torch.long)
        for block_index, start in enumerate(noisy_frame_indices):
            end = min(start + attention_block_size, total_length)
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = ref_length + block_index * attention_block_size
            if local_attn_size == -1:
                noise_context_starts[start:end] = 0
            else:
                visible_start = max(0, ref_length + block_index * attention_block_size - local_attn_size * frame_seqlen)
                noise_context_starts[start:end] = visible_start

        # Same attention_mask as constructor
        def attention_mask(q_idx, kv_idx):
            ref_mask = (q_idx < ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_ref_visible = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx < ref_length)
            clean_clean = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx >= ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_mask = clean_ref_visible | clean_clean
            noisy_ref_visible = (q_idx >= clean_ends) & (kv_idx < ref_length)
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx >= ref_length) & (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (noisy_ref_visible | C1 | C2)
            eye_mask = (q_idx == kv_idx)
            return eye_mask | ref_mask | clean_mask | noise_mask

        # Build block-level matrix by sampling first token of each block
        ref_blocks = list(range(0, ref_length, ref_tokens_per_frame)) if ref_length > 0 else []
        clean_blocks = list(range(ref_length, ref_length + video_length, frame_seqlen))
        noisy_blocks = list(range(ref_length + video_length, total_length, frame_seqlen))
        all_blocks = ref_blocks + clean_blocks + noisy_blocks
        num_blocks = len(all_blocks)

        mask_matrix = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)
        for qi, q_start in enumerate(all_blocks):
            for ki, k_start in enumerate(all_blocks):
                mask_matrix[qi, ki] = attention_mask(q_start, k_start)

        # Print
        labels = []
        for i, start in enumerate(all_blocks):
            if start < ref_length:
                labels.append(f"R{i}")
            elif start < ref_length + video_length:
                idx = (start - ref_length) // frame_seqlen
                labels.append(f"C{idx}")
            else:
                idx = (start - ref_length - video_length) // frame_seqlen
                labels.append(f"N{idx}")

        col_width = max(len(l) for l in labels) + 1
        print("=" * 80)
        print(f"[Mask Debug: with_ref] ref={num_ref_frames}x{ref_tokens_per_frame}, "
              f"clean={num_frames}x{frame_seqlen}, noisy={num_frames}x{frame_seqlen}")
        print(f"  total_tokens={total_length}, blocks={num_blocks}, "
              f"block_size={attention_block_size}, local_attn={local_attn_size}")
        header = " " * col_width + "".join(l.rjust(col_width) for l in labels)
        print(header)
        for qi, label in enumerate(labels):
            row_str = label.ljust(col_width)
            for ki in range(num_blocks):
                cell = "✓" if mask_matrix[qi, ki] else "·"
                row_str += cell.rjust(col_width)
            print(row_str)
        print("=" * 80)

    @staticmethod
    def _debug_print_mask_with_ref_i2v(
        num_ref_frames: int,
        ref_tokens_per_frame: int,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        device: torch.device | str = "cpu",
    ):
        """Print mask matrix matching _prepare_teacher_forcing_mask_with_ref_i2v logic.

        Layout: [ref(K) | clean_frame0 | clean_blocks | noisy_frame0 | noisy_blocks]
        """
        if ref_tokens_per_frame is None:
            ref_tokens_per_frame = frame_seqlen
        ref_length = num_ref_frames * ref_tokens_per_frame
        video_length = num_frames * frame_seqlen
        total_length = ref_length + 2 * video_length
        attention_block_size = frame_seqlen * num_frame_per_block
        clean_ends = ref_length + video_length

        context_ends = torch.zeros(total_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length, device=device, dtype=torch.long)

        # Ref blocks
        if ref_length > 0:
            ref_block_starts = torch.arange(0, ref_length, step=ref_tokens_per_frame, device=device, dtype=torch.long)
            for start in ref_block_starts:
                end = min(start + ref_tokens_per_frame, ref_length)
                context_ends[start:end] = ref_length
                context_starts[start:end] = 0

        # Clean first frame: only attend to ref + itself
        ff_start = ref_length
        ff_end = ref_length + frame_seqlen
        context_ends[ff_start:ff_end] = ff_end
        context_starts[ff_start:ff_end] = 0

        # Remaining clean frames
        remaining_clean_starts = torch.arange(ff_end, ref_length + video_length, step=attention_block_size, device=device, dtype=torch.long)
        for start in remaining_clean_starts:
            end = min(start + attention_block_size, ref_length + video_length)
            context_ends[start:end] = end
            if local_attn_size == -1:
                context_starts[start:end] = 0
            else:
                visible_start = max(0, end - local_attn_size * frame_seqlen)
                context_starts[start:end] = visible_start

        # Noisy first frame
        noisy_ff_start = ref_length + video_length
        noisy_ff_end = noisy_ff_start + frame_seqlen
        noise_noise_starts[noisy_ff_start:noisy_ff_end] = noisy_ff_start
        noise_noise_ends[noisy_ff_start:noisy_ff_end] = noisy_ff_end
        noise_context_starts[noisy_ff_start:noisy_ff_end] = 0
        noise_context_ends[noisy_ff_start:noisy_ff_end] = 0

        # Remaining noisy blocks
        noisy_block_starts = torch.arange(
            noisy_ff_end, ref_length + 2 * video_length, step=attention_block_size, device=device, dtype=torch.long)
        for block_index, start in enumerate(noisy_block_starts):
            end = min(start + attention_block_size, ref_length + 2 * video_length)
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            clean_visible_end = ff_end + block_index * attention_block_size
            noise_context_ends[start:end] = clean_visible_end
            if local_attn_size == -1:
                noise_context_starts[start:end] = 0
            else:
                visible_start = max(0, clean_visible_end - local_attn_size * frame_seqlen)
                noise_context_starts[start:end] = visible_start

        # Same attention_mask as constructor
        def attention_mask(q_idx, kv_idx):
            ref_mask = (q_idx < ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_ref_visible = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx < ref_length)
            clean_clean = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx >= ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_mask = clean_ref_visible | clean_clean
            noisy_ref_visible = (q_idx >= clean_ends) & (kv_idx < ref_length)
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx >= ref_length) & (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (noisy_ref_visible | C1 | C2)
            eye_mask = (q_idx == kv_idx)
            return eye_mask | ref_mask | clean_mask | noise_mask

        # Build block-level matrix
        ref_blocks = list(range(0, ref_length, ref_tokens_per_frame)) if ref_length > 0 else []
        # For i2v: first frame is separate, then remaining blocks
        clean_ff_blocks = [ref_length]
        remaining_clean = list(range(ff_end, ref_length + video_length, frame_seqlen))
        clean_blocks = clean_ff_blocks + remaining_clean
        noisy_ff_blocks = [ref_length + video_length]
        remaining_noisy = list(range(noisy_ff_end, total_length, frame_seqlen))
        noisy_blocks = noisy_ff_blocks + remaining_noisy
        all_blocks = ref_blocks + clean_blocks + noisy_blocks
        num_blocks = len(all_blocks)

        mask_matrix = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)
        for qi, q_start in enumerate(all_blocks):
            for ki, k_start in enumerate(all_blocks):
                mask_matrix[qi, ki] = attention_mask(q_start, k_start)

        # Labels: R0..Rk, CF0, C0..Cn, NF0, N0..Nn
        labels = []
        for i, start in enumerate(all_blocks):
            if start < ref_length:
                labels.append(f"R{i}")
            elif start == ref_length:
                labels.append("CF0")
            elif start < ref_length + video_length:
                idx = (start - ff_end) // frame_seqlen
                labels.append(f"C{idx}")
            elif start == ref_length + video_length:
                labels.append("NF0")
            else:
                idx = (start - noisy_ff_end) // frame_seqlen
                labels.append(f"N{idx}")

        col_width = max(len(l) for l in labels) + 1
        print("=" * 80)
        print(f"[Mask Debug: with_ref_i2v] ref={num_ref_frames}x{ref_tokens_per_frame}, "
              f"frames={num_frames}x{frame_seqlen}")
        print(f"  total_tokens={total_length}, blocks={num_blocks}, "
              f"block_size={attention_block_size}, local_attn={local_attn_size}")
        header = " " * col_width + "".join(l.rjust(col_width) for l in labels)
        print(header)
        for qi, label in enumerate(labels):
            row_str = label.ljust(col_width)
            for ki in range(num_blocks):
                cell = "✓" if mask_matrix[qi, ki] else "·"
                row_str += cell.rjust(col_width)
            print(row_str)
        print("=" * 80)

    @staticmethod
    def _prepare_teacher_forcing_mask_with_ref(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_ref_frames: int = 0,
        ref_tokens_per_frame: int = None,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """TF mask with ref prefix. Layout: [ref(K) | clean(F) | noisy(F)].

        ref_tokens_per_frame: tokens per ref frame (may differ from video frame_seqlen
                              when ref images have different spatial resolution).
                              Defaults to frame_seqlen if not specified.
        """
        if ref_tokens_per_frame is None:
            ref_tokens_per_frame = frame_seqlen
        ref_length = num_ref_frames * ref_tokens_per_frame
        video_length = num_frames * frame_seqlen
        total_length = ref_length + 2 * video_length
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        attention_block_size = frame_seqlen * num_frame_per_block
        clean_ends = ref_length + video_length

        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Ref blocks: full attention within ref (step by ref_tokens_per_frame)
        if ref_length > 0:
            ref_block_starts = torch.arange(0, ref_length, step=ref_tokens_per_frame, device=device, dtype=torch.long)
            for start in ref_block_starts:
                end = min(start + ref_tokens_per_frame, ref_length)
                context_ends[start:end] = ref_length
                context_starts[start:end] = 0

        # Clean blocks: offset by ref_length, attend to all ref + preceding clean
        clean_frame_indices = torch.arange(ref_length, ref_length + video_length, step=attention_block_size, device=device, dtype=torch.long)
        for start in clean_frame_indices:
            end = min(start + attention_block_size, ref_length + video_length)
            context_ends[start:end] = end
            if local_attn_size == -1:
                context_starts[start:end] = 0
            else:
                visible_start = max(0, end - local_attn_size * frame_seqlen)
                context_starts[start:end] = visible_start

        # Noisy blocks: attend to own block + corresponding clean context + all ref
        noisy_start = ref_length + video_length
        noisy_frame_indices = torch.arange(noisy_start, total_length, step=attention_block_size, device=device, dtype=torch.long)
        for block_index, start in enumerate(noisy_frame_indices):
            end = min(start + attention_block_size, total_length)
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = ref_length + block_index * attention_block_size
            if local_attn_size == -1:
                noise_context_starts[start:end] = 0
            else:
                visible_start = max(0, ref_length + block_index * attention_block_size - local_attn_size * frame_seqlen)
                noise_context_starts[start:end] = visible_start

        def attention_mask(b, h, q_idx, kv_idx):
            ref_mask = (q_idx < ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_ref_visible = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx < ref_length)
            clean_clean = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx >= ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_mask = clean_ref_visible | clean_clean
            noisy_ref_visible = (q_idx >= clean_ends) & (kv_idx < ref_length)
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx >= ref_length) & (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (noisy_ref_visible | C1 | C2)
            eye_mask = (q_idx == kv_idx)
            return eye_mask | ref_mask | clean_mask | noise_mask

        return create_block_mask(attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length,
            _compile=True, device=device)

    @staticmethod
    def _prepare_teacher_forcing_mask_with_ref_i2v_bk(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_ref_frames: int = 0,
        ref_tokens_per_frame: int = None,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """TF mask with ref prefix + independent first frame.
        Layout: [ref(K) | clean_frame0 | clean_blocks | noisy_frame0 | noisy_blocks].

        ref_tokens_per_frame: tokens per ref frame (may differ from video frame_seqlen).
                              Defaults to frame_seqlen if not specified.
        """
        if ref_tokens_per_frame is None:
            ref_tokens_per_frame = frame_seqlen
        ref_length = num_ref_frames * ref_tokens_per_frame
        video_length = num_frames * frame_seqlen
        total_length = ref_length + 2 * video_length
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        attention_block_size = frame_seqlen * num_frame_per_block
        clean_ends = ref_length + video_length

        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Ref blocks: full attention within ref (step by ref_tokens_per_frame)
        if ref_length > 0:
            ref_block_starts = torch.arange(0, ref_length, step=ref_tokens_per_frame, device=device, dtype=torch.long)
            for start in ref_block_starts:
                end = min(start + ref_tokens_per_frame, ref_length)
                context_ends[start:end] = ref_length
                context_starts[start:end] = 0

        # Clean first frame: only attend to ref + itself
        ff_start = ref_length
        ff_end = ref_length + frame_seqlen
        context_ends[ff_start:ff_end] = ff_end
        context_starts[ff_start:ff_end] = 0

        # Remaining clean frames: blockwise from ff_end onwards
        remaining_clean_starts = torch.arange(ff_end, ref_length + video_length, step=attention_block_size, device=device, dtype=torch.long)
        for start in remaining_clean_starts:
            end = min(start + attention_block_size, ref_length + video_length)
            context_ends[start:end] = end
            if local_attn_size == -1:
                context_starts[start:end] = 0
            else:
                visible_start = max(0, end - local_attn_size * frame_seqlen)
                context_starts[start:end] = visible_start

        # Noisy first frame
        noisy_ff_start = ref_length + video_length
        noisy_ff_end = noisy_ff_start + frame_seqlen
        noise_noise_starts[noisy_ff_start:noisy_ff_end] = noisy_ff_start
        noise_noise_ends[noisy_ff_start:noisy_ff_end] = noisy_ff_end
        noise_context_starts[noisy_ff_start:noisy_ff_end] = 0
        noise_context_ends[noisy_ff_start:noisy_ff_end] = 0

        # Remaining noisy blocks
        noisy_block_starts = torch.arange(
            noisy_ff_end, ref_length + 2 * video_length, step=attention_block_size, device=device, dtype=torch.long)
        for block_index, start in enumerate(noisy_block_starts):
            end = min(start + attention_block_size, ref_length + 2 * video_length)
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            clean_visible_end = ff_end + block_index * attention_block_size
            noise_context_ends[start:end] = clean_visible_end
            if local_attn_size == -1:
                noise_context_starts[start:end] = 0
            else:
                visible_start = max(0, clean_visible_end - local_attn_size * frame_seqlen)
                noise_context_starts[start:end] = visible_start

        def attention_mask(b, h, q_idx, kv_idx):
            ref_mask = (q_idx < ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_ref_visible = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx < ref_length)
            clean_clean = (q_idx >= ref_length) & (q_idx < clean_ends) & (kv_idx >= ref_length) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_starts[q_idx])
            clean_mask = clean_ref_visible | clean_clean
            noisy_ref_visible = (q_idx >= clean_ends) & (kv_idx < ref_length)
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx >= ref_length) & (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (noisy_ref_visible | C1 | C2)
            eye_mask = (q_idx == kv_idx)
            return eye_mask | ref_mask | clean_mask | noise_mask

        return create_block_mask(attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length,
            _compile=True, device=device)

    @staticmethod
    def _prepare_teacher_forcing_mask_with_ref_i2v(
        device,
        num_frames,
        frame_seqlen,
        ref_token_len,
        num_frame_per_block=1,
        independent_first_frame=False,
        local_attn_size=-1,
    ):
        ref_length = int(ref_token_len)
        video_length = num_frames * frame_seqlen
        total_length = ref_length + 2 * video_length
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        seq_len = total_length + padded_length

        abs_size = frame_seqlen * num_frame_per_block       # attention_block_size
        clean_ends = ref_length + video_length
        local_window = local_attn_size * frame_seqlen
        use_local = local_attn_size != -1

        if not independent_first_frame:
            # ---------- uniform blockwise ----------
            def attention_mask(b, h, q_idx, kv_idx):
                eye = q_idx == kv_idx
                ref_mask = (q_idx < ref_length) & (kv_idx < ref_length)
                kv_is_ref = kv_idx < ref_length

                # clean: 可见自身 local window + 永远可见 ref image
                in_clean = (q_idx >= ref_length) & (q_idx < clean_ends)
                kc = (q_idx - ref_length) // abs_size
                c_bstart = ref_length + kc * abs_size
                c_bend = torch.clamp(c_bstart + abs_size, max=clean_ends)
                if use_local:
                    c_start = torch.clamp(c_bend - local_window, min=0)
                else:
                    c_start = torch.zeros_like(q_idx)
                clean_local = (kv_idx < c_bend) & (kv_idx >= c_start)
                clean_mask = in_clean & (clean_local | kv_is_ref)

                # noisy: 自己 block + 对应 clean 前缀 + 永远可见 ref image
                in_noisy = (q_idx >= clean_ends) & (q_idx < total_length)
                kn = (q_idx - clean_ends) // abs_size            # block_index
                n_bstart = clean_ends + kn * abs_size
                n_bend = torch.clamp(n_bstart + abs_size, max=total_length)
                nc_end = ref_length + kn * abs_size
                if use_local:
                    nc_start = torch.clamp(nc_end - local_window, min=0)
                else:
                    nc_start = torch.zeros_like(q_idx)
                C1 = (kv_idx < n_bend) & (kv_idx >= n_bstart)
                C2 = (kv_idx < nc_end) & (kv_idx >= nc_start)
                noise_mask = in_noisy & (C1 | C2 | kv_is_ref)

                return eye | ref_mask | clean_mask | noise_mask

        else:
            # ---------- independent first frame ----------
            ff_end = ref_length + frame_seqlen
            noisy_ff_start = clean_ends
            noisy_ff_end = clean_ends + frame_seqlen

            def attention_mask(b, h, q_idx, kv_idx):
                eye = q_idx == kv_idx
                ref_mask = (q_idx < ref_length) & (kv_idx < ref_length)
                kv_is_ref = kv_idx < ref_length

                # clean first frame: 见 [0, ff_end)，已包含 ref
                ff_mask = (q_idx >= ref_length) & (q_idx < ff_end) & (kv_idx < ff_end)

                # remaining clean blocks: local window + 永远可见 ref
                in_rem_clean = (q_idx >= ff_end) & (q_idx < clean_ends)
                kc = (q_idx - ff_end) // abs_size
                c_bstart = ff_end + kc * abs_size
                c_bend = torch.clamp(c_bstart + abs_size, max=clean_ends)
                if use_local:
                    c_start = torch.clamp(c_bend - local_window, min=0)
                else:
                    c_start = torch.zeros_like(q_idx)
                clean_local = (kv_idx < c_bend) & (kv_idx >= c_start)
                rem_clean_mask = in_rem_clean & (clean_local | kv_is_ref)

                clean_mask = ff_mask | rem_clean_mask

                # noisy first frame: 自己 frame + 永远可见 ref
                nff_mask = (q_idx >= noisy_ff_start) & (q_idx < noisy_ff_end) \
                    & ((kv_idx >= noisy_ff_start) & (kv_idx < noisy_ff_end) | kv_is_ref)

                # remaining noisy blocks: 自己 block + 对应 clean 前缀 + 永远可见 ref
                in_rem_noisy = (q_idx >= noisy_ff_end) & (q_idx < total_length)
                bi = (q_idx - noisy_ff_end) // abs_size          # block_index
                n_bstart = noisy_ff_end + bi * abs_size
                n_bend = torch.clamp(n_bstart + abs_size, max=total_length)
                clean_vis_end = ff_end + bi * abs_size
                if use_local:
                    nc_start = torch.clamp(clean_vis_end - local_window, min=0)
                else:
                    nc_start = torch.zeros_like(q_idx)
                C1 = (kv_idx < n_bend) & (kv_idx >= n_bstart)
                C2 = (kv_idx < clean_vis_end) & (kv_idx >= nc_start)
                rem_noise_mask = in_rem_noisy & (C1 | C2 | kv_is_ref)

                noise_mask = nff_mask | rem_noise_mask

                return eye | ref_mask | clean_mask | noise_mask

        return create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=seq_len, KV_LEN=seq_len,
            _compile=True, device=device,
        )

    @staticmethod
    def _print_frame_level_teacher_forcing_mask(
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        independent_first_frame: bool = False,
        device="cpu",
    ):
        """
        在帧级别打印 teacher forcing mask，不调用 to_dense，避免长序列爆显存。

        对每一帧只取其起始 token 作为代表来判断可见性，结果与真实 mask 一致。
        序列布局: [clean: 帧0..帧N-1] [noisy: 帧0..帧N-1]

        Args:
            num_frames: latent 帧数。
            frame_seqlen: 每帧的 token 数。
            num_frame_per_block: 每个 attention block 含多少帧（首帧独立时不含首帧）。
            local_attn_size: 局部注意力窗口大小（以帧为单位），-1 表示全局。
            independent_first_frame: True 走 i2v 布局（首帧独立成块），False 走均匀分块。
            device: 构建辅助数组的设备，建议 "cpu" 以完全不占显存。
        """

        total_length = num_frames * frame_seqlen * 2
        clean_ends_val = num_frames * frame_seqlen
        attention_block_size = frame_seqlen * num_frame_per_block

        context_ends = torch.zeros(total_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length, device=device, dtype=torch.long)

        # ===== clean 部分 =====
        if independent_first_frame:
            # i2v: 首帧独立，只能 attend 自己
            context_ends[:frame_seqlen] = frame_seqlen
            context_starts[:frame_seqlen] = 0
            clean_block_start = frame_seqlen
        else:
            clean_block_start = 0

        remaining_clean_indices = torch.arange(
            clean_block_start, num_frames * frame_seqlen,
            attention_block_size, device=device, dtype=torch.long
        )
        for start in remaining_clean_indices:
            s = int(start)
            context_ends[s:s + attention_block_size] = s + attention_block_size
            if local_attn_size == -1:
                context_starts[s:s + attention_block_size] = 0
            else:
                context_starts[s:s + attention_block_size] = max(
                    0, s + attention_block_size - local_attn_size * frame_seqlen
                )

        # ===== noisy 部分 =====
        noisy_offset = num_frames * frame_seqlen

        if independent_first_frame:
            # i2v: noisy 首帧独立，attend clean 首帧 + 自身
            nf_s, nf_e = noisy_offset, noisy_offset + frame_seqlen
            noise_noise_starts[nf_s:nf_e] = nf_s
            noise_noise_ends[nf_s:nf_e] = nf_e
            noise_context_starts[nf_s:nf_e] = 0
            noise_context_ends[nf_s:nf_e] = 0  # 首帧前无 clean context
            noisy_block_start = noisy_offset + frame_seqlen
            # i2v 下，第 block_index 个 noisy block 可见的 clean 末端含独立首帧
            clean_visible_end_base = frame_seqlen
        else:
            noisy_block_start = noisy_offset
            clean_visible_end_base = 0

        noisy_block_starts = torch.arange(
            noisy_block_start, noisy_offset + num_frames * frame_seqlen,
            attention_block_size, device=device, dtype=torch.long
        )
        for block_index, start in enumerate(noisy_block_starts):
            s = int(start)
            e = s + attention_block_size
            noise_noise_starts[s:e] = s
            noise_noise_ends[s:e] = e
            clean_visible_end = clean_visible_end_base + block_index * attention_block_size
            noise_context_ends[s:e] = clean_visible_end
            if local_attn_size == -1:
                noise_context_starts[s:e] = 0
            else:
                noise_context_starts[s:e] = max(
                    0, clean_visible_end - local_attn_size * frame_seqlen
                )

        def visible(q_idx, kv_idx):
            clean_mask = (
                (q_idx < clean_ends_val)
                and (kv_idx < context_ends[q_idx])
                and (kv_idx >= context_starts[q_idx])
            )
            C1 = (kv_idx < noise_noise_ends[q_idx]) and (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) and (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends_val) and (C1 or C2)
            eye_mask = (q_idx == kv_idx)
            return bool(eye_mask or clean_mask or noise_mask)

        # 每帧取起始 token 作为代表
        q_reps = [(f"c{i}", i * frame_seqlen) for i in range(num_frames)] + \
                [(f"n{i}", clean_ends_val + i * frame_seqlen) for i in range(num_frames)]

        mode = "i2v (independent first frame)" if independent_first_frame else "uniform blockwise"
        print(f"===== Frame-level Teacher Forcing Mask [{mode}] =====")
        print(f"num_frames={num_frames}, frame_seqlen={frame_seqlen}, "
            f"num_frame_per_block={num_frame_per_block}, local_attn_size={local_attn_size}")
        print("(row = query frame, col = key/value frame; ✓ = visible)")

        labels = [lbl for lbl, _ in q_reps]
        header = "       " + " ".join(f"{l:>3}" for l in labels)
        print(header)
        for q_lbl, q_pos in q_reps:
            row = " ".join(
                f"{'✓' if visible(q_pos, k_pos) else '·':>3}"
                for _, k_pos in q_reps
            )
            print(f"{q_lbl:>5}  {row}")
        print("=" * len(header))

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
        act_context_scale=1.0,
        ref_latents=None,
        ref_mask=None,
        cross_attn_frame_ranges=None,
        cross_attn_num_frames=None,
        ref_only_prefill: bool = False,
    ):
        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if seq_len is None:
            seq_len = x.shape[2] * x.shape[-2] * x.shape[-1] // (
                self.patch_size[1] * self.patch_size[2] * self.patch_size[0]
            )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = self.add_action_control(x, act_context, act_context_scale)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x, dim=0)

        # ---- Process ref latents (same as _forward_train) ----
        has_ref = ref_latents is not None and ref_mask is not None
        num_ref_frames = 0
        ref_tokens_per_frame = None
        N_r = 0
        ref_tokens = None
        rH, rW = None, None
        if has_ref:
            if ref_latents.ndim == 6:
                ref_latents = ref_latents.squeeze(3)
            num_ref_frames = ref_latents.shape[1]
            H_ref_lat = ref_latents.shape[3]
            W_ref_lat = ref_latents.shape[4]
            ref_tokens_per_frame = (H_ref_lat // self.patch_size[1]) * (W_ref_lat // self.patch_size[2])

            B = x.shape[0]
            K = ref_latents.shape[1]
            C_ref = ref_latents.shape[2]
            ref_flat = ref_latents.reshape(B * K, C_ref, 1, H_ref_lat, W_ref_lat).to(device=device, dtype=x.dtype)
            target_c = self.in_dim
            if C_ref < target_c:
                ref_flat = torch.nn.functional.pad(ref_flat, (0, 0, 0, 0, 0, 0, 0, target_c - C_ref))
            elif C_ref > target_c:
                ref_flat = ref_flat[:, :target_c]
            ref_embedded = self.patch_embedding(ref_flat)
            rH, rW = ref_embedded.shape[3], ref_embedded.shape[4]
            tokens_per_slot = rH * rW
            ref_embedded = ref_embedded.reshape(B, K, self.dim, tokens_per_slot)
            ref_embedded = ref_embedded.permute(0, 1, 3, 2).reshape(B, K * tokens_per_slot, self.dim)
            m = ref_mask.to(device=device, dtype=x.dtype).view(B, K, 1, 1).expand(B, K, tokens_per_slot, 1).reshape(B, K * tokens_per_slot, 1)
            ref_tokens = ref_embedded * m
            N_r = K * tokens_per_slot
            # Prepend ref tokens to sequence
            x = torch.cat([ref_tokens, x], dim=1)
            # print(f"[RefImage Inference] N_r={N_r}, tokens_per_slot={tokens_per_slot}, K={K}")

        # Compute frame_seqlen: video tokens per frame (from seq_lens and num_frames)
        # At this point x is [B, seq_len, dim], so frame_seqlen = video_seq_len / num_frames
        num_video_frames = t.shape[1]
        # video_seq_len before ref tokens prepend = seq_lens[0].item() (or seq_len)
        video_seq_len = seq_lens[0].item()
        frame_seqlen = video_seq_len // num_video_frames

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)  # [B, num_frames, 6, dim]

        # Ref time embedding: t=0 for all ref frames
        e0_ref = None
        if ref_tokens is not None:
            B = x.shape[0]
            K = num_ref_frames
            t_ref = torch.zeros(B, K, dtype=t.dtype, device=device)
            e_ref = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_ref.flatten()).to(x.dtype)
            )
            e0_ref = self.time_projection(e_ref).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=(B, K))
            e0 = torch.cat([e0_ref, e0], dim=1)

        if ref_only_prefill:
            if ref_tokens is None or N_r == 0:
                raise ValueError("ref-only KV prefill requires ref_latents/ref_mask")
            # The dummy video frame was needed only to reuse the normal i2v
            # patch-embedding setup.  It must not enter any transformer block.
            x = ref_tokens
            e0 = e0_ref
            seq_lens = torch.full(
                (x.shape[0],), N_r, dtype=torch.long, device=device
            )

        context_lens = None
        context = self._embed_text_context(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = self._append_clip_context(context, context_clip)

        cross_attn_chunk_ids = None
        if ref_only_prefill and context.ndim == 4:
            # Ref tokens are the prefix of the teacher sequence and do not
            # participate in chunk-wise prompt cross-attention.
            cross_attn_chunk_ids = torch.full(
                x.shape[:2], -1, dtype=torch.long, device=x.device
            )
        elif context.ndim == 4:
            raw_num_frames = resolve_cross_attn_num_frames(
                cross_attn_num_frames, batch_size=context.shape[0]
            )
            total_latent_frames = 1 + math.ceil(max(raw_num_frames - 1, 0) / 4)
            latent_frame_offset = int(current_start) // frame_seqlen
            prefix_tokens = x.shape[1] - video_seq_len
            cross_attn_chunk_ids = self._build_chunk_routes(
                context,
                cross_attn_frame_ranges,
                raw_num_frames,
                latent_frames=num_video_frames,
                spatial_tokens_per_frame=frame_seqlen,
                prefix_tokens=prefix_tokens,
                latent_frame_offset=latent_frame_offset,
                total_latent_frames=total_latent_frames,
            )
            if cross_attn_chunk_ids.shape != x.shape[:2]:
                raise RuntimeError(
                    "chunk route shape does not match inference tokens: "
                    f"routes={tuple(cross_attn_chunk_ids.shape)}, x={tuple(x.shape)}"
                )
        elif cross_attn_frame_ranges is not None:
            raise ValueError(
                "cross_attn_frame_ranges were provided for non-chunkwise prompt embeddings"
            )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            num_ref_tokens=N_r,
            num_ref_frames=num_ref_frames if has_ref else 0,
            ref_tokens_per_frame=ref_tokens_per_frame,
            cross_attn_chunk_ids=cross_attn_chunk_ids,
            ref_only_prefill=ref_only_prefill,
        )

        # Set ref flags for self-attention modules (same as _forward_train)
        for block_index, block in enumerate(self.blocks):
            block.self_attn._is_teacher_forcing = False
            if has_ref and ref_tokens_per_frame is not None:
                block.self_attn._num_ref_tokens = N_r
                block.self_attn._ref_tokens_per_frame = ref_tokens_per_frame
                block.self_attn._ref_grid_sizes = (rH, rW)
                kv_cache[block_index]["_ref_tokens_per_frame"] = int(
                    ref_tokens_per_frame
                )
                kv_cache[block_index]["_ref_grid_sizes"] = (int(rH), int(rW))
            elif int(kv_cache[block_index].get("_num_ref_tokens", 0)) == 0:
                block.self_attn._num_ref_tokens = 0
                block.self_attn._ref_tokens_per_frame = None
                block.self_attn._ref_grid_sizes = None
            else:
                # Video-only denoising follows a ref-only prefill.  Keep the
                # cached ref metadata instead of resetting it with N_r=0.
                block.self_attn._num_ref_tokens = int(
                    kv_cache[block_index]["_num_ref_tokens"]
                )
                block.self_attn._ref_tokens_per_frame = kv_cache[block_index].get(
                    "_ref_tokens_per_frame"
                )
                block.self_attn._ref_grid_sizes = kv_cache[block_index].get(
                    "_ref_grid_sizes"
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

        if ref_only_prefill:
            return None

        # Strip ref tokens before head/unpatchify
        if N_r > 0:
            x = x[:, N_r:]

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def add_action_control(self, x, act_context, act_context_scale=1.0):
        if isinstance(act_context_scale, (int, float)) and act_context_scale == 0.0:
            return x
        if act_context is not None and self.act_control_adapter is not None:
            x_new = []
            # print(f"control adapter add act control,act_context.shape:{act_context.shape} act_context_scale: {act_context_scale}")
            with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, "_train_dmd_action_adapter", False)):
                y_action = [self.act_control_adapter(u.unsqueeze(0)) for u in act_context]
            # Ensure act_context_scale is on the same device
            if isinstance(act_context_scale, torch.Tensor):
                act_context_scale = act_context_scale.to(device=x[0].device, dtype=x[0].dtype)
            for u, v in zip(x, y_action):
                t_f = u.shape[2]
                c_f = v.shape[2]
                if t_f > c_f:
                    offset = t_f - c_f
                    u = torch.cat([u[:, :offset], u[:, offset:] + v * act_context_scale], dim=1)
                else:
                    u = u + v * act_context_scale
                x_new.append(u)
            x = x_new
        return x

    def _forward_train(
        self,
        x, #b c f h w(1, 16, 21, h, w)
        t,
        context,
        seq_len,
        y=None,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        act_context=None,
        act_context_scale=0.0,
        ref_latents=None,
        ref_mask=None,
        cross_attn_frame_ranges=None,
        cross_attn_num_frames=None,
        current_start=0,
    ):
        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if seq_len is None:
            seq_len = x.shape[2] * x.shape[-2] * x.shape[-1] // (
                self.patch_size[1] * self.patch_size[2] * self.patch_size[0]
            )

        # Compute ref info before block_mask construction
        has_ref = ref_latents is not None and ref_mask is not None
        num_ref_frames = 0
        ref_tokens_per_frame = None
        if has_ref:
            if ref_latents.ndim == 6:
                assert ref_latents.dim() == 6, "Expected ref_latents shape [B, N_r, C, 1, h, w]"
                assert ref_latents.shape[3] == 1, "Expected ref_latents to have a single frame (shape[3] == 1)"
                ref_latents = ref_latents.squeeze(3)  # Remove the singleton frame dimension, new shape: [B, K, C, h, w]
            num_ref_frames = ref_latents.shape[1]  # K slots
            # Compute ref tokens per frame from actual ref latent spatial dims.
            # ref_latents shape: [B, K, C, H_ref, W_ref]
            # After patch_embedding with patch_size (1, ph, pw):
            #   tokens_per_ref_frame = (H_ref // ph) * (W_ref // pw)
            # This may differ from video frame_seqlen if ref images have
            # different spatial resolution.
            H_ref_lat = ref_latents.shape[3]
            W_ref_lat = ref_latents.shape[4]
            ref_tokens_per_frame = (H_ref_lat // self.patch_size[1]) * (W_ref_lat // self.patch_size[2])
            print(f"[RefImage] num_ref_frames={num_ref_frames}, ref_mask.sum()={ref_mask.sum().item()}, "
                  f"ref_latent_spatial={H_ref_lat}x{W_ref_lat}, ref_tokens_per_frame={ref_tokens_per_frame}")


        if self.block_mask is None and not getattr(self, "_uses_dense_score_attention", False):
            num_frames = x.shape[2]
            frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
            print(f"[RefImage] video frame_seqlen={frame_seqlen}, ref_tokens_per_frame={ref_tokens_per_frame}")
            if clean_x is not None:
                if num_ref_frames > 0:
                    if self.independent_first_frame:
                        # self.block_mask = self._prepare_teacher_forcing_mask_with_ref_i2v(
                        #     device, num_frames=num_frames,
                        #     frame_seqlen=frame_seqlen,
                        #     num_ref_frames=num_ref_frames,
                        #     ref_tokens_per_frame=ref_tokens_per_frame,
                        #     num_frame_per_block=self.num_frame_per_block,
                        #     local_attn_size=self.local_attn_size,
                        # )
                        self.block_mask = self._prepare_teacher_forcing_mask_with_ref_i2v(
                            device=device,
                            num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            ref_token_len=num_ref_frames * ref_tokens_per_frame,
                            num_frame_per_block=self.num_frame_per_block,
                            independent_first_frame=True,
                            local_attn_size=self.local_attn_size,
                        )
                        print(f"block_mask: {self.block_mask}")
                        self._debug_print_mask_with_ref_i2v(
                            num_ref_frames=num_ref_frames,
                            ref_tokens_per_frame=ref_tokens_per_frame,
                            num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            num_frame_per_block=self.num_frame_per_block,
                            local_attn_size=self.local_attn_size,
                            device=device,
                        )
                    else:
                        self.block_mask = self._prepare_teacher_forcing_mask_with_ref(
                            device, num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            num_ref_frames=num_ref_frames,
                            ref_tokens_per_frame=ref_tokens_per_frame,
                            num_frame_per_block=self.num_frame_per_block,
                            local_attn_size=self.local_attn_size,
                        )
                        self._debug_print_mask_with_ref(
                            num_ref_frames=num_ref_frames,
                            ref_tokens_per_frame=ref_tokens_per_frame,
                            num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            num_frame_per_block=self.num_frame_per_block,
                            local_attn_size=self.local_attn_size,
                            device=device,
                        )
                else:
                    if self.independent_first_frame:
                        self.block_mask = self._prepare_teacher_forcing_mask_i2v(
                            device, num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            num_frame_per_block=self.num_frame_per_block,
                            local_attn_size=self.local_attn_size,
                        )
                        # self._print_frame_level_teacher_forcing_mask_i2v(
                        #     num_frames=num_frames,
                        #     frame_seqlen=frame_seqlen,
                        #     num_frame_per_block=self.num_frame_per_block,
                        #     local_attn_size=self.local_attn_size,
                        #     device="cpu",   # 用 cpu 构建辅助数组，完全不占 GPU 显存
                        # )
                    else:
                        self.block_mask = self._prepare_teacher_forcing_mask(
                            device, num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            num_frame_per_block=self.num_frame_per_block,
                            local_attn_size=self.local_attn_size,
                        )
                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                        # 直接构建 block 级别的 compact mask，不依赖 to_dense
                        attention_block_size = frame_seqlen * self.num_frame_per_block
                        num_blocks = num_frames // self.num_frame_per_block
                        clean_ends = num_frames * frame_seqlen
                        total_blocks = num_blocks * 2
                        local_attn = self.local_attn_size
                        print(f"local_attn={local_attn}, num_blocks={num_blocks}, frame_seqlen={frame_seqlen}, num_frame_per_block={self.num_frame_per_block}, independent_first_frame={self.independent_first_frame}")

                        compact_mask = torch.zeros(total_blocks, total_blocks, dtype=torch.int32)
                        for qi in range(total_blocks):
                            for ki in range(total_blocks):
                                if qi < num_blocks:
                                    q_pos = qi * attention_block_size
                                else:
                                    q_pos = clean_ends + (qi - num_blocks) * attention_block_size

                                if ki < num_blocks:
                                    k_pos = ki * attention_block_size
                                else:
                                    k_pos = clean_ends + (ki - num_blocks) * attention_block_size

                                is_clean_q = q_pos < clean_ends
                                if is_clean_q:
                                    # clean→clean: k_pos 在 [context_starts, context_ends) 内
                                    block_end = (qi + 1) * attention_block_size
                                    if local_attn == -1:
                                        visible_start = 0
                                    else:
                                        visible_start = max(0, block_end - local_attn * frame_seqlen)
                                    visible = (k_pos < block_end) and (k_pos >= visible_start)
                                else:
                                    # noisy→noisy: 同 block
                                    noisy_block_idx = qi - num_blocks
                                    noisy_start = clean_ends + noisy_block_idx * attention_block_size
                                    noisy_end = noisy_start + attention_block_size
                                    same_noisy_block = (k_pos >= noisy_start) and (k_pos < noisy_end)
                                    # noisy→clean: k_pos 在 [noise_context_starts, noise_context_ends) 内
                                    noise_ctx_end = noisy_block_idx * attention_block_size
                                    if local_attn == -1:
                                        noise_ctx_start = 0
                                    else:
                                        noise_ctx_start = max(0, noise_ctx_end - local_attn * frame_seqlen)
                                    see_clean = (k_pos < clean_ends) and (k_pos < noise_ctx_end) and (k_pos >= noise_ctx_start)
                                    visible = same_noisy_block or see_clean

                                compact_mask[qi, ki] = int(visible)

                        # 打印
                        block_labels = [f"c{i}" for i in range(num_blocks)] + [f"n{i}" for i in range(num_blocks)]
                        header = "      " + " ".join(f"{l:>3}" for l in block_labels)
                        print(header)
                        for i, label in enumerate(block_labels):
                            row = " ".join(f"{'✓' if compact_mask[i,j] else '·':>3}" for j in range(total_blocks))
                            print(f"{label:>5} {row}")
                    # 帧级打印（不调用 to_dense，辅助数组放 cpu 不占显存）
                    self._print_frame_level_teacher_forcing_mask(
                        num_frames=num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                        independent_first_frame=self.independent_first_frame,
                        device="cpu",
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        #x[0] shape: torch.Size([32, 21, 60, 104]), x[0] dtype: torch.float32, y[0] shape: torch.Size([16, 21, 60, 104]), y[0] dtype: torch.float32
        # print(f"x[0] shape: {x[0].shape}, x[0] dtype: {x[0].dtype}, y[0] shape: {y[0].shape}, y[0] dtype: {y[0].dtype}") 
        print(f"x[0] shape: {x[0].shape}, x[0] dtype: {x[0].dtype}") 

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = self.add_action_control(x, act_context, act_context_scale)

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

        # ---- Process ref latents ----
        ref_tokens = None
        N_r = 0
        if has_ref:
            B = x.shape[0]
            K = ref_latents.shape[1]
            C_ref = ref_latents.shape[2]
            H_ref, W_ref = ref_latents.shape[3], ref_latents.shape[4]
            ref_flat = ref_latents.reshape(B * K, C_ref, 1, H_ref, W_ref).to(device=device, dtype=x.dtype)
            # Channel alignment
            target_c = self.in_dim
            assert C_ref == self.in_dim, f"C_ref={C_ref}, self.in_dim={self.in_dim}"
            if C_ref < target_c:
                ref_flat = torch.nn.functional.pad(ref_flat, (0, 0, 0, 0, 0, 0, 0, target_c - C_ref))
            elif C_ref > target_c:
                ref_flat = ref_flat[:, :target_c]
            ref_embedded = self.patch_embedding(ref_flat)  # [B*K, dim, 1, H', W']
            rH, rW = ref_embedded.shape[3], ref_embedded.shape[4]
            tokens_per_slot = rH * rW
            ref_embedded = ref_embedded.reshape(B, K, self.dim, tokens_per_slot)
            ref_embedded = ref_embedded.permute(0, 1, 3, 2).reshape(B, K * tokens_per_slot, self.dim)
            # Apply mask: zero out tokens for dropped ref images
            m = ref_mask.to(device=device, dtype=x.dtype).view(B, K, 1, 1).expand(B, K, tokens_per_slot, 1).reshape(B, K * tokens_per_slot, 1)
            ref_tokens = ref_embedded * m
            N_r = K * tokens_per_slot
            # Note: grid_sizes keeps original video frame count for RoPE
            # (ref frames use separate ref_grid in self-attention)
            print(f"[RefImage] N_r={N_r}, tokens_per_slot={tokens_per_slot}, K={K}")

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype)
        ) # 1, 21, 768
        print(f"e shape: {e.shape}") #e shape: torch.Size([21, 1536])
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape) # [B, num_frames, 6, dim]
        print(f"e0 shape: {e0.shape}") #e0 shape: torch.Size([1, 21, 6, 1536])

        context_lens = None
        context = self._embed_text_context(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = self._append_clip_context(context, context_clip)

        if clean_x is not None:
            if y is not None:
                clean_x = [torch.cat([u, v], dim=0) for u, v in zip(clean_x, y)]

            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = self.add_action_control(clean_x, act_context, act_context_scale)

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

            # Concatenate: [ref, clean, noisy]
            if ref_tokens is not None:
                x = torch.cat([ref_tokens, clean_x, x], dim=1)
                print(f"ref_tokens.shape={ref_tokens.shape}, clean_x.shape={clean_x.shape}, x.shape={x.shape}")
            else:
                x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).to(x.dtype)
            )
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

            # Ref time embedding: t=0 for all ref frames
            if ref_tokens is not None:
                B = x.shape[0]
                K = num_ref_frames
                t_ref = torch.zeros(B, K, dtype=t.dtype, device=device)
                e_ref = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t_ref.flatten()).to(x.dtype)
                )
                e0_ref = self.time_projection(e_ref).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=(B, K))
                print(f"e0_ref.shape={e0_ref.shape}, e0.shape={e0.shape}, e0_clean.shape={e0_clean.shape}") 
                assert len(e0_ref.shape) == len(e0.shape) == len(e0_clean.shape) == 4, f"e0_ref.shape={e0_ref.shape}, ref_tokens.shape={e0.shape}, e0_clean.shape={e0_clean.shape}"
                e0 = torch.cat([e0_ref, e0_clean, e0], dim=1)
            else:
                e0 = torch.cat([e0_clean, e0], dim=1)
        else:
            # No TF: optionally prepend ref tokens
            if ref_tokens is not None:
                B = x.shape[0]
                K = num_ref_frames
                x = torch.cat([ref_tokens, x], dim=1)
                t_ref = torch.zeros(B, K, dtype=t.dtype, device=device)
                e_ref = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t_ref.flatten()).to(x.dtype)
                )
                e0_ref = self.time_projection(e_ref).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=(B, K))
                e0 = torch.cat([e0_ref, e0], dim=1)

        video_latent_frames = int(grid_sizes[0, 0].item())
        spatial_tokens_per_frame = int(
            grid_sizes[0, 1].item() * grid_sizes[0, 2].item()
        )
        routed_num_frames = (
            resolve_cross_attn_num_frames(
                cross_attn_num_frames, batch_size=context.shape[0]
            )
            if context.ndim == 4
            else None
        )
        total_routed_latent_frames = (
            1 + math.ceil(max(routed_num_frames - 1, 0) / 4)
            if routed_num_frames is not None
            else None
        )
        cross_attn_chunk_ids = self._build_chunk_routes(
            context,
            cross_attn_frame_ranges,
            routed_num_frames,
            latent_frames=video_latent_frames,
            spatial_tokens_per_frame=spatial_tokens_per_frame,
            prefix_tokens=N_r,
            video_copies=2 if clean_x is not None else 1,
            total_latent_frames=total_routed_latent_frames,
            latent_frame_offset=0,
        )
        if cross_attn_chunk_ids is not None and cross_attn_chunk_ids.shape != x.shape[:2]:
            raise RuntimeError(
                "chunk route shape does not match training tokens: "
                f"routes={tuple(cross_attn_chunk_ids.shape)}, x={tuple(x.shape)}"
            )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            num_ref_tokens=N_r,
            num_ref_frames=num_ref_frames if has_ref else 0,
            ref_tokens_per_frame=ref_tokens_per_frame,
            cross_attn_chunk_ids=cross_attn_chunk_ids,
        )

        # Set TF and ref flags for self-attention modules
        is_tf_mode = clean_x is not None
        for block in self.blocks:
            block.self_attn._is_teacher_forcing = is_tf_mode
            block.self_attn._num_ref_tokens = N_r
            block.self_attn._ref_tokens_per_frame = ref_tokens_per_frame
            if has_ref and ref_tokens_per_frame is not None:
                block.self_attn._ref_grid_sizes = (rH, rW)
            else:
                block.self_attn._ref_grid_sizes = None

        def create_custom_forward(module):
            def custom_forward(*inputs, **kw):
                return module(*inputs, **kw)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs, use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # Strip ref tokens first, then split clean/noisy
        if N_r > 0:
            x = x[:, N_r:]
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # Use original grid_sizes and e for head/unpatchify (video frames only)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

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
