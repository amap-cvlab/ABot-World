from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .attention import grouped_query_attention
from .config import FlashDecoderConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return (x_float * self.weight.float()).to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _axis_rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Apply RoPE to x [..., sequence, axis_dim]."""
    dim = x.shape[-1]
    if dim == 0:
        return x
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(x.dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def apply_3d_rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Apply factorized temporal/height/width RoPE.

    x is [B, heads, L, head_dim], positions is [L, 3].  The head
    dimension is split 1/4 temporal and 3/8 for each spatial axis.
    """
    head_dim = x.shape[-1]
    t_dim = head_dim // 4
    remaining = head_dim - t_dim
    h_dim = (remaining // 4) * 2
    # Keep every rotary section even; XL resolves to 16/24/24.
    w_dim = head_dim - t_dim - h_dim
    xt, xh, xw = x.split((t_dim, h_dim, w_dim), dim=-1)
    return torch.cat(
        (
            _axis_rope(xt, positions[:, 0], theta),
            _axis_rope(xh, positions[:, 1], theta),
            _axis_rope(xw, positions[:, 2], theta),
        ),
        dim=-1,
    )


@dataclass
class LayerCache:
    key: Tensor
    value: Tensor
    positions: Tensor
    groups: Optional[Tensor] = None


@dataclass
class StreamingState:
    backbone: list[Optional[LayerCache]]
    refinement: list[Optional[LayerCache]]
    latent_frames_seen: int = 0


class GQAAttention(nn.Module):
    def __init__(self, config: FlashDecoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.v_norm = RMSNorm(self.head_dim) if config.kv_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: Optional[LayerCache],
        cache_tokens: int,
        groups: Optional[Tensor] = None,
        causal_window: Optional[int] = None,
    ) -> tuple[Tensor, LayerCache]:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        q = apply_3d_rope(q, positions, self.theta)

        if cache is not None:
            # Rebase cached frames to negative temporal positions.  Keeping
            # unrotated keys in the cache makes relative-window RoPE exact and
            # prevents positions from growing beyond the training range.
            past_positions = cache.positions.clone()
            past_positions[:, 0] -= int(past_positions[:, 0].max().item()) + 1
            k_all = torch.cat((cache.key, k), dim=2)
            v_all = torch.cat((cache.value, v), dim=2)
            pos_all = torch.cat((past_positions, positions), dim=0)
            if groups is not None:
                if cache.groups is None:
                    raise ValueError("causal chunk attention requires cached group ids")
                past_groups = cache.groups.clone()
                past_groups -= int(past_groups.max().item()) + 1
                groups_all = torch.cat((past_groups, groups), dim=0)
            else:
                groups_all = None
        else:
            k_all, v_all, pos_all = k, v, positions
            groups_all = groups

        k_attn = apply_3d_rope(k_all, pos_all, self.theta)
        if causal_window is None:
            attended = grouped_query_attention(q, k_attn, v_all)
        else:
            if groups is None or groups_all is None:
                raise ValueError("causal chunk attention requires group ids")
            attended_groups = []
            for group in groups.unique_consecutive():
                query_mask = groups.eq(group)
                key_distance = group - groups_all
                key_mask = key_distance.ge(0) & key_distance.le(causal_window)
                attended_groups.append(
                    grouped_query_attention(
                        q[:, :, query_mask],
                        k_attn[:, :, key_mask],
                        v_all[:, :, key_mask],
                    )
                )
            attended = torch.cat(attended_groups, dim=2)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)

        keep = min(cache_tokens, k_all.shape[2])
        new_cache = LayerCache(
            k_all[:, :, -keep:],
            v_all[:, :, -keep:],
            pos_all[-keep:],
            groups_all[-keep:] if groups_all is not None else None,
        )
        return self.out_proj(attended), new_cache


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: FlashDecoderConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = GQAAttention(config)
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, config.mlp_ratio)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: Optional[LayerCache],
        cache_tokens: int,
        groups: Optional[Tensor] = None,
        causal_window: Optional[int] = None,
    ) -> tuple[Tensor, LayerCache]:
        update, cache = self.attn(
            self.attn_norm(x),
            positions,
            cache,
            cache_tokens,
            groups,
            causal_window,
        )
        x = x + update
        return x + self.mlp(self.mlp_norm(x)), cache


class StreamingTransformer(nn.Module):
    def __init__(self, config: FlashDecoderConfig, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(depth))
        self.norm = RMSNorm(config.dim)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        caches: list[Optional[LayerCache]],
        cache_tokens: int,
        groups: Optional[Tensor] = None,
        causal_window: Optional[int] = None,
    ) -> tuple[Tensor, list[LayerCache]]:
        next_caches: list[LayerCache] = []
        for block, layer_cache in zip(self.blocks, caches):
            if self.gradient_checkpointing and self.training:
                if layer_cache is None:
                    def block_only(inp: Tensor, pos: Tensor, module: TransformerBlock = block) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                        out, new = module(
                            inp, pos, None, cache_tokens, groups, causal_window
                        )
                        return out, new.key, new.value, new.positions, new.groups

                    x, key, value, cache_positions, cache_groups = checkpoint(
                        block_only, x, positions, use_reentrant=False
                    )
                else:
                    def block_cached(
                        inp: Tensor,
                        pos: Tensor,
                        key_in: Tensor,
                        value_in: Tensor,
                        positions_in: Tensor,
                        groups_in: Tensor,
                        module: TransformerBlock = block,
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                        cached = LayerCache(key_in, value_in, positions_in, groups_in)
                        out, new = module(
                            inp, pos, cached, cache_tokens, groups, causal_window
                        )
                        return out, new.key, new.value, new.positions, new.groups

                    if layer_cache.groups is None:
                        raise ValueError("streaming cache is missing group ids")
                    x, key, value, cache_positions, cache_groups = checkpoint(
                        block_cached,
                        x,
                        positions,
                        layer_cache.key,
                        layer_cache.value,
                        layer_cache.positions,
                        layer_cache.groups,
                        use_reentrant=False,
                    )
                layer_cache = LayerCache(key, value, cache_positions, cache_groups)
            else:
                x, layer_cache = block(
                    x,
                    positions,
                    layer_cache,
                    cache_tokens,
                    groups,
                    causal_window,
                )
            next_caches.append(layer_cache)
        return self.norm(x), next_caches


class SpatialHead(nn.Module):
    def __init__(self, config: FlashDecoderConfig) -> None:
        super().__init__()
        hidden = int(config.dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, config.out_channels * config.spatial_factor**2, bias=True)
        self.factor = config.spatial_factor
        self.out_channels = config.out_channels

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, frames, tokens, _ = x.shape
        x = self.fc2(F.silu(self.fc1(x)))
        x = x.view(batch * frames, height, width, -1).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.factor)
        return x.view(batch, frames, self.out_channels, height * self.factor, width * self.factor)


class FlashDecoder(nn.Module):
    def __init__(self, config: FlashDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_proj = nn.Linear(config.latent_channels, config.dim)
        self.backbone = StreamingTransformer(config, config.depth)
        self.temporal_up = nn.Linear(config.dim, config.dim * config.temporal_factor)
        self.refinement = StreamingTransformer(config, config.refinement_depth)
        self.spatial_head = SpatialHead(config)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Residual projections use depth-aware initialization.
        scale = 0.02 / (2 * max(self.config.depth, 1)) ** 0.5
        for transformer in (self.backbone, self.refinement):
            for block in transformer.blocks:
                nn.init.normal_(block.attn.out_proj.weight, std=scale)
                nn.init.normal_(block.mlp.down.weight, std=scale)

    def empty_state(self) -> StreamingState:
        return StreamingState(
            backbone=[None] * len(self.backbone.blocks),
            refinement=[None] * len(self.refinement.blocks),
            latent_frames_seen=0,
        )

    @staticmethod
    def _positions(frames: int, height: int, width: int, device: torch.device) -> Tensor:
        t = torch.arange(frames, device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        grid = torch.meshgrid(t, h, w, indexing="ij")
        return torch.stack(grid, dim=-1).reshape(-1, 3)

    def stream_chunk(
        self,
        latent_chunk: Tensor,
        state: StreamingState,
        causal_within_chunk: bool = False,
    ) -> tuple[Tensor, StreamingState]:
        """Decode one or more latent frames and update the rolling caches.

        Wan's causal temporal convention is asymmetric: the first latent
        represents one startup pixel frame and every later latent represents
        ``rt`` frames.  The startup frame uses the final expanded temporal
        slot. Frames inside one call attend bidirectionally; separate calls
        remain streaming through the fixed-size rolling cache.
        """
        if latent_chunk.ndim != 5:
            raise ValueError("latent_chunk must have shape [B, C, T, H, W]")
        batch, channels, frames, height, width = latent_chunk.shape
        if channels != self.config.latent_channels:
            raise ValueError(f"expected {self.config.latent_channels} latent channels, got {channels}")
        if frames < 1:
            raise ValueError("latent_chunk must contain at least one frame")
        tokens_per_frame = height * width
        x = latent_chunk.permute(0, 2, 3, 4, 1).reshape(
            batch, frames * tokens_per_frame, channels
        )
        x = self.latent_proj(x)
        positions = self._positions(frames, height, width, x.device)
        backbone_groups = torch.arange(frames, device=x.device).repeat_interleave(
            tokens_per_frame
        )
        x, backbone_cache = self.backbone(
            x,
            positions,
            state.backbone,
            self.config.window_frames * tokens_per_frame,
            backbone_groups,
            self.config.window_frames if causal_within_chunk else None,
        )

        rt = self.config.temporal_factor
        x = self.temporal_up(x).view(
            batch, frames, tokens_per_frame, rt, self.config.dim
        )
        expanded = []
        expanded_groups = []
        for index in range(frames):
            frame = x[:, index].permute(0, 2, 1, 3)
            latent_index = state.latent_frames_seen + index
            if (
                self.config.temporal_layout == "h3"
                and latent_index % self.config.temporal_chunk_size == 0
            ):
                trim = self.config.temporal_chunk_trim
                frame = frame[:, trim:]
            elif self.config.temporal_layout == "wan" and latent_index == 0:
                frame = frame[:, -1:]
            expanded.append(frame)
            expanded_groups.extend([index] * frame.shape[1])
        x = torch.cat(expanded, dim=1)
        expanded_frames = x.shape[1]
        refine_positions = self._positions(expanded_frames, height, width, x.device)
        refinement_groups = torch.tensor(
            expanded_groups, device=x.device
        ).repeat_interleave(tokens_per_frame)
        x = x.reshape(batch, expanded_frames * tokens_per_frame, self.config.dim)
        x, refinement_cache = self.refinement(
            x,
            refine_positions,
            state.refinement,
            self.config.window_frames * rt * tokens_per_frame,
            refinement_groups,
            self.config.window_frames if causal_within_chunk else None,
        )
        x = x.view(batch, expanded_frames, tokens_per_frame, self.config.dim)
        pixels = self.spatial_head(x, height, width)
        return pixels, StreamingState(
            backbone_cache,
            refinement_cache,
            latent_frames_seen=state.latent_frames_seen + frames,
        )

    def stream_step(self, latent_frame: Tensor, state: StreamingState) -> tuple[Tensor, StreamingState]:
        """Decode one latent frame using the streaming chunk implementation."""
        if latent_frame.ndim != 4:
            raise ValueError("latent_frame must have shape [B, C, H, W]")
        return self.stream_chunk(latent_frame.unsqueeze(2), state)

    def forward(
        self,
        latents: Tensor,
        output_frames: Optional[int] = None,
        latent_chunk_size: int = 1,
        latent_chunk_causal: bool = False,
    ) -> Tensor:
        """Decode latent frames using the configured VAE temporal layout."""
        if latents.ndim != 5:
            raise ValueError("latents must have shape [B, C, T, H, W]")
        if latent_chunk_size < 1:
            raise ValueError("latent_chunk_size must be positive")
        state = self.empty_state()
        chunks = []
        for start in range(0, latents.shape[2], latent_chunk_size):
            pixels, state = self.stream_chunk(
                latents[:, :, start : start + latent_chunk_size],
                state,
                causal_within_chunk=latent_chunk_causal,
            )
            chunks.append(pixels)
        output = torch.cat(chunks, dim=1).permute(0, 2, 1, 3, 4).contiguous()
        if output_frames is not None:
            output = output[:, :, :output_frames]
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
