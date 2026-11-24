"""Utilities for routing chunk-wise text prompts to Wan video tokens."""

from __future__ import annotations

import math

import torch


def resolve_cross_attn_num_frames(value, *, batch_size: int | None = None) -> int:
    """Return one shared raw-frame count from a scalar or collated tensor."""

    if isinstance(value, torch.Tensor):
        values = value.detach().reshape(-1).to(dtype=torch.long)
        if values.numel() == 0:
            raise ValueError("cross_attn_num_frames cannot be empty")
        if batch_size is not None and values.numel() not in (1, int(batch_size)):
            raise ValueError(
                "cross_attn_num_frames must be scalar or have one value per batch "
                f"item, got {values.numel()} values for batch_size={batch_size}"
            )
        if not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError(
                "all samples in a batch must use the same cross_attn_num_frames"
            )
        value = values[0].item()
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("cross_attn_num_frames cannot be empty")
        values = [int(item) for item in value]
        if any(item != values[0] for item in values[1:]):
            raise ValueError(
                "all samples in a batch must use the same cross_attn_num_frames"
            )
        value = values[0]

    value = int(value)
    if value <= 0:
        raise ValueError("cross_attn_num_frames must be positive")
    return value


def build_chunkwise_cross_attn_ids(
    frame_ranges: torch.Tensor,
    *,
    num_frames: int,
    latent_frames: int,
    spatial_tokens_per_frame: int,
    prefix_tokens: int = 0,
    latent_frame_offset: int = 0,
    total_latent_frames: int | None = None,
    device=None,
) -> torch.Tensor:
    """Map raw-frame prompt ranges onto flattened causal-VAE video tokens.

    Wan's first temporal latent represents raw frame 0. Every following latent
    represents a causal group of source frames (normally four). Each latent is
    assigned to the prompt chunk with maximum overlap. ``latent_frame_offset``
    supports block-wise KV-cache forwards, while ``prefix_tokens`` assigns -1
    to reference-image tokens so they cannot read any local prompt chunk.
    """

    frame_ranges = torch.as_tensor(frame_ranges, dtype=torch.long, device=device)
    if frame_ranges.ndim == 2:
        frame_ranges = frame_ranges.unsqueeze(0)
    if frame_ranges.ndim != 3 or frame_ranges.shape[-1] != 2:
        raise ValueError(
            f"frame_ranges must have shape [B,C,2], got {tuple(frame_ranges.shape)}"
        )

    num_frames = int(num_frames)
    latent_frames = int(latent_frames)
    spatial_tokens_per_frame = int(spatial_tokens_per_frame)
    prefix_tokens = int(prefix_tokens)
    latent_frame_offset = int(latent_frame_offset)
    if total_latent_frames is None:
        total_latent_frames = latent_frame_offset + latent_frames
    total_latent_frames = int(total_latent_frames)
    if (
        num_frames <= 0
        or latent_frames <= 0
        or spatial_tokens_per_frame <= 0
        or prefix_tokens < 0
        or latent_frame_offset < 0
        or total_latent_frames < latent_frame_offset + latent_frames
    ):
        raise ValueError("invalid dimensions for chunk-wise cross-attention routing")

    invalid_ranges = (
        (frame_ranges[..., 0] < 0)
        | (frame_ranges[..., 1] <= frame_ranges[..., 0])
        | (frame_ranges[..., 1] > num_frames)
    )
    if invalid_ranges.any().item():
        raise ValueError(f"invalid chunk frame range: {frame_ranges.tolist()}")

    temporal_stride = (
        1
        if total_latent_frames == 1
        else max(1, math.ceil((num_frames - 1) / (total_latent_frames - 1)))
    )
    latent_indices = torch.arange(
        latent_frame_offset,
        latent_frame_offset + latent_frames,
        dtype=torch.long,
        device=frame_ranges.device,
    )
    latent_starts = 1 + (latent_indices - 1).clamp_min(0) * temporal_stride
    latent_starts = torch.where(
        latent_indices == 0, torch.zeros_like(latent_starts), latent_starts
    )
    latent_ends = torch.minimum(
        latent_starts + temporal_stride,
        latent_starts.new_full((latent_frames,), num_frames),
    )
    latent_ends = torch.where(
        latent_indices == 0, torch.ones_like(latent_ends), latent_ends
    )

    overlap = (
        torch.minimum(latent_ends[None, :, None], frame_ranges[:, None, :, 1])
        - torch.maximum(latent_starts[None, :, None], frame_ranges[:, None, :, 0])
    ).clamp_min(0)
    best_overlap, latent_routes = overlap.max(dim=-1)
    if (best_overlap <= 0).any().item():
        bad_batch, bad_local_latent = torch.nonzero(
            best_overlap <= 0, as_tuple=False
        )[0].tolist()
        raise ValueError(
            f"no chunk prompt covers batch {bad_batch} latent "
            f"{latent_frame_offset + bad_local_latent} source frames "
            f"[{latent_starts[bad_local_latent].item()}, "
            f"{latent_ends[bad_local_latent].item()})"
        )

    latent_routes = latent_routes.repeat_interleave(
        spatial_tokens_per_frame, dim=1
    )
    prefix = torch.full(
        (frame_ranges.shape[0], prefix_tokens),
        -1,
        dtype=torch.long,
        device=frame_ranges.device,
    )
    return torch.cat([prefix, latent_routes], dim=1)
