from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FlashDecoderConfig:
    latent_channels: int = 48
    out_channels: int = 3
    dim: int = 1536
    depth: int = 20
    refinement_depth: int = 2
    num_heads: int = 24
    num_kv_heads: int = 3
    mlp_ratio: float = 4.0
    temporal_factor: int = 4
    temporal_layout: str = "wan"
    temporal_chunk_size: int = 5
    temporal_chunk_trim: int = 0
    spatial_factor: int = 16
    window_frames: int = 2
    rope_theta: float = 10_000.0
    qk_norm: bool = True
    kv_norm: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 8:
            raise ValueError("head_dim must be divisible by 8 for 3D RoPE")
        if self.temporal_factor < 1 or self.spatial_factor < 1:
            raise ValueError("upsampling factors must be positive")
        if self.temporal_layout not in {"wan", "h3"}:
            raise ValueError("temporal_layout must be 'wan' or 'h3'")
        if self.temporal_chunk_size < 1:
            raise ValueError("temporal_chunk_size must be positive")
        if not 0 <= self.temporal_chunk_trim < self.temporal_factor:
            raise ValueError("temporal_chunk_trim must be in [0, temporal_factor)")
        if self.window_frames < 1:
            raise ValueError("window_frames must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FlashDecoderConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})
