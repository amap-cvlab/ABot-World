import logging
import torch
import torch._dynamo

# torch._dynamo.config.recompile_limit = 1024
# torch._dynamo.config.suppress_errors = True

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import time

__all__ = [
    "Wan2_2_VAE",
]

CACHE_T = 2


def _extract_checkpoint_state_dict(raw):
    state = raw
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "gen_model" in state:
        state = state["gen_model"]
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like state_dict.")
    return state


def _map_lightvae_key_to_wanvae(key):
    def _map_resnet_tail(tail):
        if tail.startswith("norm1."):
            return "residual.0." + tail[len("norm1."):]
        if tail.startswith("conv1."):
            return "residual.2." + tail[len("conv1."):]
        if tail.startswith("norm2."):
            return "residual.3." + tail[len("norm2."):]
        if tail.startswith("conv2."):
            return "residual.6." + tail[len("conv2."):]
        if tail.startswith("conv_shortcut."):
            return "shortcut." + tail[len("conv_shortcut."):]
        return tail

    # Skip training-only projection heads.
    if key.startswith("dynamic_feature_projection_heads."):
        return None

    # Top-level projections.
    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "conv1.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "conv2.", 1)

    # Encoder direct blocks.
    if key.startswith("encoder.conv_in."):
        return key.replace("encoder.conv_in.", "encoder.conv1.", 1)
    if key.startswith("encoder.mid_block.resnets.0."):
        tail = key[len("encoder.mid_block.resnets.0."):]
        return "encoder.middle.0." + _map_resnet_tail(tail)
    if key.startswith("encoder.mid_block.attentions.0."):
        return key.replace("encoder.mid_block.attentions.0.", "encoder.middle.1.", 1)
    if key.startswith("encoder.mid_block.resnets.1."):
        tail = key[len("encoder.mid_block.resnets.1."):]
        return "encoder.middle.2." + _map_resnet_tail(tail)
    if key.startswith("encoder.norm_out."):
        return key.replace("encoder.norm_out.", "encoder.head.0.", 1)
    if key.startswith("encoder.conv_out."):
        return key.replace("encoder.conv_out.", "encoder.head.2.", 1)

    # Encoder down blocks.
    if key.startswith("encoder.down_blocks."):
        parts = key.split(".")
        # encoder.down_blocks.{i}.resnets.{j}.*
        if len(parts) >= 6 and parts[3] == "resnets":
            tail = ".".join(parts[5:])
            return f"encoder.downsamples.{parts[2]}.downsamples.{parts[4]}." + _map_resnet_tail(tail)
        # encoder.down_blocks.{i}.downsampler.resample.1.*
        if len(parts) >= 7 and parts[3] == "downsampler" and parts[4] == "resample":
            return f"encoder.downsamples.{parts[2]}.downsamples.2.resample.{parts[5]}." + ".".join(parts[6:])
        # encoder.down_blocks.{i}.downsampler.time_conv.*
        if len(parts) >= 6 and parts[3] == "downsampler" and parts[4] == "time_conv":
            return f"encoder.downsamples.{parts[2]}.downsamples.2.time_conv." + ".".join(parts[5:])

    # Decoder direct blocks.
    if key.startswith("decoder.conv_in."):
        return key.replace("decoder.conv_in.", "decoder.conv1.", 1)
    if key.startswith("decoder.mid_block.resnets.0."):
        tail = key[len("decoder.mid_block.resnets.0."):]
        return "decoder.middle.0." + _map_resnet_tail(tail)
    if key.startswith("decoder.mid_block.attentions.0."):
        return key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.", 1)
    if key.startswith("decoder.mid_block.resnets.1."):
        tail = key[len("decoder.mid_block.resnets.1."):]
        return "decoder.middle.2." + _map_resnet_tail(tail)
    if key.startswith("decoder.norm_out."):
        return key.replace("decoder.norm_out.", "decoder.head.0.", 1)
    if key.startswith("decoder.conv_out."):
        return key.replace("decoder.conv_out.", "decoder.head.2.", 1)

    # Decoder up blocks.
    if key.startswith("decoder.up_blocks."):
        parts = key.split(".")
        # decoder.up_blocks.{i}.resnets.{j}.*
        if len(parts) >= 6 and parts[3] == "resnets":
            tail = ".".join(parts[5:])
            return f"decoder.upsamples.{parts[2]}.upsamples.{parts[4]}." + _map_resnet_tail(tail)
        # decoder.up_blocks.{i}.upsampler.resample.1.*
        if len(parts) >= 7 and parts[3] == "upsampler" and parts[4] == "resample":
            return f"decoder.upsamples.{parts[2]}.upsamples.3.resample.{parts[5]}." + ".".join(parts[6:])
        # decoder.up_blocks.{i}.upsampler.time_conv.*
        if len(parts) >= 6 and parts[3] == "upsampler" and parts[4] == "time_conv":
            return f"decoder.upsamples.{parts[2]}.upsamples.3.time_conv." + ".".join(parts[5:])

    # If already in wan naming, keep it.
    return key


def _normalize_vae_state_dict(raw_state):
    state = _extract_checkpoint_state_dict(raw_state)
    norm = {}
    for k, v in state.items():
        nk = _map_lightvae_key_to_wanvae(k)
        if nk is None:
            continue
        norm[nk] = v
    return norm


def infer_lightvae_pruning_rate_from_ckpt(vae_pth, full_decoder_conv1_out=1024):
    """
    Infer LightVAE pruning rate from decoder conv1 out-channels in checkpoint.
    For Wan2.2 VAE decoder, full (unpruned) decoder.conv1 out-channels is 1024.
    """
    if vae_pth is None or not os.path.exists(vae_pth):
        return None
    try:
        raw_state = torch.load(vae_pth, map_location="cpu")
        state = _extract_checkpoint_state_dict(raw_state)
    except Exception as e:
        logging.warning(f"Failed to load checkpoint for pruning-rate inference: {e}")
        return None

    weight = None
    if isinstance(state, dict):
        if "decoder.conv_in.weight" in state:
            weight = state["decoder.conv_in.weight"]
        elif "decoder.conv1.weight" in state:
            weight = state["decoder.conv1.weight"]

    if weight is None:
        try:
            norm_state = _normalize_vae_state_dict(state)
            weight = norm_state.get("decoder.conv1.weight", None)
        except Exception:
            weight = None
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) < 1:
        return None

    student_out = int(weight.shape[0])
    if full_decoder_conv1_out <= 0:
        return None
    pruning_rate = 1.0 - (float(student_out) / float(full_decoder_conv1_out))
    # keep within reasonable range and stable text representation
    pruning_rate = max(0.0, min(0.99, pruning_rate))
    return round(pruning_rate, 6)


def convert_to_channels_last_3d(module):
    """
    Recursively convert all Conv3d weights in module to channels_last_3d format.
    This eliminates NCHW<->NHWC format conversion overhead in cuDNN.
    """
    for child in module.children():
        if isinstance(child, nn.Conv3d):
            child.weight.data = child.weight.data.to(memory_format=torch.channels_last_3d)
        else:
            convert_to_channels_last_3d(child)


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        dims = (1 if self.channel_first else -1)
        # Use a more compiler-friendly RMS implementation
        rms = (x.pow(2).mean(dims, keepdim=True) + 1e-6).sqrt()
        return (x / rms) * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]

                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"

                cache_x = x[:, :, -CACHE_T:, :, :].clone()

                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    if cache_x.shape[2] < 2:
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(x.device),
                            cache_x
                        ], dim=2)
                    x = self.time_conv(x, feat_cache[idx])

                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)

                if first_chunk:
                    x = x[:, :, 1:, :, :]

        t_now = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t_now)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

        return x

    def init_weight(self, conv):
        conv_weight = conv.weight.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else nn.Identity())

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x).reshape(b * t, 1, c * 3,
                                   -1).permute(0, 1, 3,
                                               2).contiguous().chunk(3, dim=-1))

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class AvgDown3D(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            factor_t,
            factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            factor_t,
            factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_downsample=False,
                 down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,  # 3
                 temperal_upsample=False,
                 up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):  # 3
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False, profiler=None):
        x_main = x.clone()
        for i, module in enumerate(self.upsamples):
            x_main = module(x_main, feat_cache, feat_idx, first_chunk)

        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):

    def __init__(
            self,
            dim=128,
            z_dim=4,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i]
                if i < len(temperal_downsample) else False)
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                ))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):

    def __init__(
            self,
            dim=128,
            z_dim=4,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_upsample=[False, True, True],
            dropout=0.0,
    ):
        super().__init__()
        self.dim = dim  # 256
        self.z_dim = z_dim  # 48
        self.dim_mult = dim_mult  # [1, 2, 4, 4]
        self.num_res_blocks = num_res_blocks  # 2
        self.attn_scales = attn_scales  # []
        self.temperal_upsample = temperal_upsample  # [True, True, False]

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]  # [1024, 1024, 1024, 512, 256]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)  # 0.25
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # temperal_upsample = [True, True, False]
            t_up_flag = temperal_upsample[i] if i < len(
                temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,  # 3
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,  # dim_mult = [1, 2, 4, 4]
                ))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False, profiler=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # 1. Middle Blocks
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # 2. Upsample Blocks
        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        # 3. Head
        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(
            self,
            dim=160,
            dec_dim=256,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            # temperal_downsample=[True, True, False],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            pruning_rate=0.0,
    ):
        super().__init__()
        self.dim = dim  # 160
        self.z_dim = z_dim  # 48
        self.dim_mult = dim_mult  # [1, 2, 4, 4]
        self.num_res_blocks = num_res_blocks  # 2
        self.attn_scales = attn_scales  # []
        self.temperal_downsample = temperal_downsample  # [False, True, True]
        self.temperal_upsample = temperal_downsample[::-1]  # [True, True, False]

        # Pruning-compatible with Turbo-VAED LightVAE student.
        dim = max(1, int(round(dim * (1.0 - pruning_rate))))
        dec_dim = max(1, int(round(dec_dim * (1.0 - pruning_rate))))

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def forward(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        """Like decode() but preserves feat_cache across calls for streaming."""
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        first_chunk = self._feat_map[0] is None
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=first_chunk,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=16, dim=160, device="cpu", **kwargs):
    # params
    cfg = dict(
        dim=dim,  # 160
        z_dim=z_dim,  # 48
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],  # [False, True, True]
        dropout=0.0,
    )
    cfg.update(**kwargs)

    if device == "meta":
        with torch.device("meta"):
            model = WanVAE_(**cfg)
    else:
        model = WanVAE_(**cfg)

    # load checkpoint
    if pretrained_path is not None and os.path.exists(pretrained_path):
        logging.info(f"Wan2.2 VAE loading {pretrained_path}")
        raw_state = torch.load(pretrained_path, map_location="cpu")
        state_dict = _normalize_vae_state_dict(raw_state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        logging.info(
            f"VAE checkpoint loaded with strict=False (missing={len(missing)}, unexpected={len(unexpected)})"
        )

        # Convert Conv3d weights to channels_last_3d for cuDNN optimization
        convert_to_channels_last_3d(model)
        logging.info("VAE: Converted Conv3d weights to channels_last_3d format")
    else:
        error_msg = f"VAE checkpoint not found at {pretrained_path}!"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    return model


class Wan2_2_VAE:

    def __init__(
            self,
            z_dim=48,
            c_dim=160,
            vae_pth=None,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
            dtype=torch.float,
            device="cuda",
            vae_type="wan2.2",
            lightvae_pruning_rate=None,
            lightvae_encoder_vae_pth="/root/kaichen/Wan2.2_VAE.pth",
    ):

        self.dtype = dtype
        self.device = device
        self.vae_type = vae_type
        self.encoder_model = None

        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        self.scale = [mean, 1.0 / std]

        # init model
        if self.vae_type == "wan2.2":
            self.model = (
                _video_vae(
                    pretrained_path=vae_pth,
                    z_dim=z_dim,  # 48
                    dim=c_dim,  # 160
                    dim_mult=dim_mult,  # [1, 2, 4, 4]
                    temperal_downsample=temperal_downsample,  # [False, True, True]
                ).eval().requires_grad_(False).to(device=device, dtype=dtype))
        elif self.vae_type == "mg_lightvae":
            resolved_pruning_rate = lightvae_pruning_rate
            if resolved_pruning_rate is None:
                resolved_pruning_rate = infer_lightvae_pruning_rate_from_ckpt(vae_pth)
                if resolved_pruning_rate is None:
                    resolved_pruning_rate = 0.75
                    logging.warning(
                        "Unable to infer LightVAE pruning rate from checkpoint; fallback to 0.75."
                    )
            logging.info(
                f"Loading mg_lightvae decoder from {vae_pth} (pruning_rate={resolved_pruning_rate}), "
                f"while keeping teacher encoder from {lightvae_encoder_vae_pth}."
            )
            # Teacher encoder branch (for conditioning latents): standard Wan2.2 VAE.
            self.encoder_model = (
                _video_vae(
                    pretrained_path=lightvae_encoder_vae_pth,
                    z_dim=z_dim,
                    dim=c_dim,
                    dim_mult=dim_mult,
                    temperal_downsample=temperal_downsample,
                    pruning_rate=0.0,
                ).eval().requires_grad_(False).to(device=device, dtype=dtype)
            )
            # Student decoder branch (for reconstruction): pruned LightVAE checkpoint.
            self.model = (
                _video_vae(
                    pretrained_path=vae_pth,
                    z_dim=z_dim,
                    dim=c_dim,
                    dim_mult=dim_mult,
                    temperal_downsample=temperal_downsample,
                    pruning_rate=resolved_pruning_rate,
                ).eval().requires_grad_(False).to(device=device, dtype=dtype))
        else:
            raise ValueError(f"Unsupported vae_type: {self.vae_type}")

    def encode(self, videos):
        try:
            if not isinstance(videos, list):
                raise TypeError("videos should be a list")
            encode_model = self.encoder_model if self.vae_type == "mg_lightvae" and self.encoder_model is not None else self.model
            return [
                encode_model.encode(
                    u.unsqueeze(0).to(device=self.device, dtype=self.dtype),
                    self.scale,
                ).squeeze(0)
                for u in videos
            ]
        except TypeError as e:
            logging.info(e)
            return None

    def decode(self, zs):
        try:
            if not isinstance(zs, list):
                raise TypeError("zs should be a list")
            return [
                self.model.decode(u.unsqueeze(0).to(device=self.device, dtype=self.dtype),
                                  self.scale).clamp_(-1,
                                                     1).squeeze(0)
                for u in zs
            ]
        except TypeError as e:
            logging.info(e)
            return None

    def _decode_body(self, z, feat_cache, first_chunk=False, segment_size=5, profiler=None):
        # 1. Denormalize latents
        t_prep = time.time()
        mean, inv_std = self.scale[0], self.scale[1]
        if isinstance(mean, torch.Tensor):
            z = z / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        else:
            z = z / inv_std + mean
        if profiler is not None:
            torch.cuda.synchronize()
            profiler['vae_prep'] = profiler.get('vae_prep', 0) + (time.time() - t_prep)

        t_conv2 = time.time()
        x = self.model.conv2(z)
        if profiler is not None:
            torch.cuda.synchronize()
            profiler['vae_conv2'] = profiler.get('vae_conv2', 0) + (time.time() - t_conv2)

        iter_ = x.shape[2]
        segment_outputs = []
        t_loop = time.time()
        for i in range(0, iter_, segment_size):
            current_feat_idx = [0]
            end_i = min(i + segment_size, iter_)
            x_segment = x[:, :, i:end_i, :, :]
            chunk_out = self.model.decoder(
                x_segment,
                feat_cache=feat_cache,
                feat_idx=current_feat_idx,
                first_chunk=(first_chunk if i == 0 else False),
                profiler=profiler
            )
            segment_outputs.append(chunk_out)
        out = segment_outputs[0] if len(segment_outputs) == 1 else torch.cat(segment_outputs, dim=2)
        if profiler is not None:
            torch.cuda.synchronize()
            profiler['vae_decoder_loop'] = profiler.get('vae_decoder_loop', 0) + (time.time() - t_loop)

        t_post = time.time()
        out = unpatchify(out, patch_size=2)
        out = out.clamp_(-1, 1)
        if profiler is not None:
            torch.cuda.synchronize()
            profiler['vae_post'] = profiler.get('vae_post', 0) + (time.time() - t_post)
        return out

    def stream_decode(self, z, feat_cache, first_chunk=False, segment_size=5, profiler=None, compile_decoder=False):
        """
        Stream decode video latents using feature cache for temporal consistency.

        Args:
            z (torch.Tensor): Input latents of shape [B, C, T, H, W].
            feat_cache (list): List of cached features from previous chunks.
            first_chunk (bool): Whether this is the first chunk of a video.
            profiler (dict, optional): Dictionary to store timing information.
            compile_decoder (bool): Whether to trigger torch.compile on the decoder.

        Returns:
            out (torch.Tensor): Decoded video frames.
            feat_cache (list): Updated feature cache.
        """
        if compile_decoder and hasattr(self.model, "decoder") and not hasattr(self.model.decoder, "_is_compiled"):
            logging.info("Triggering torch.compile on VAE Decoder (Static Mode)...")
            self.model.decoder = torch.compile(
                self.model.decoder,
                dynamic=False,
                fullgraph=False
            )
            self.model.decoder._is_compiled = True

        try:
            out = self._decode_body(
                z,
                feat_cache,
                first_chunk=first_chunk,
                segment_size=segment_size,
                profiler=profiler,
            )
            return out, feat_cache

        except Exception as e:
            logging.error(f"Error in stream_decode: {e}")
            return None, feat_cache
