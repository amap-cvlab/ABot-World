import types
from pathlib import Path
from typing import List, Optional
import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors_file

from abot_world.iris.utils.scheduler import SchedulerInterface, FlowMatchScheduler
import os


def _wan_models_path(*parts) -> str:
    """Resolve wan_models path relative to project root (works with symlink and any cwd)."""
    root = Path(__file__).resolve().parents[3]
    return str((root / "wan_models").joinpath(Path(*parts)))


def _resolve_wan_path(path: str) -> str:
    """If path starts with wan_models/, resolve to absolute path (project root); else return as-is."""
    if path and path.startswith("wan_models/"):
        return _wan_models_path(path[len("wan_models/"):])
    return path


def _resolve_wan_path_with_dir(path: str, wan_models_dir: Optional[str] = None) -> str:
    """Resolve path: if wan_models_dir is set and path starts with wan_models/, use wan_models_dir as base; else _resolve_wan_path."""
    if not path:
        return path
    if wan_models_dir and path.startswith("wan_models/"):
        return os.path.join(wan_models_dir, path[len("wan_models/"):])
    return _resolve_wan_path(path)


def model_kwargs_with_relative_rope(args, default: bool = False) -> dict:
    """Merge top-level use_relative_rope into model_kwargs with a stable default."""
    raw_model_kwargs = getattr(args, "model_kwargs", {}) or {}
    model_kwargs = dict(raw_model_kwargs)
    if "use_relative_rope" not in model_kwargs:
        try:
            model_kwargs["use_relative_rope"] = bool(getattr(args, "use_relative_rope"))
        except Exception:
            model_kwargs["use_relative_rope"] = bool(default)
    return model_kwargs

from abot_world.iris.wan.modules.tokenizers import HuggingfaceTokenizer
from abot_world.iris.wan.modules.model import WanModel
from abot_world.iris.wan.modules.t5 import umt5_xxl
from abot_world.iris.wan.modules.causal_model import CausalWanModel

class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        tokenizer_path="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/",
        encoder_pth_path="wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    ) -> None:
        super().__init__()
        # tokenizer_path = _resolve_wan_path_with_dir(tokenizer_path, wan_models_dir)
        # encoder_pth_path = _resolve_wan_path_with_dir(encoder_pth_path, wan_models_dir)

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.bfloat16,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        state_dict = torch.load(encoder_pth_path,
                                map_location='cpu', weights_only=False)
        self.text_encoder.load_state_dict(state_dict)
        del state_dict

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str], device: torch.device = None, max_length: int = None) -> dict:
        tokenizer_kwargs = {}
        if max_length is not None:
            if type(max_length) is not int or max_length <= 0:
                raise ValueError("max_length must be a positive integer")
            tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": max_length}
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True, **tokenizer_kwargs)
        # When DynamicSwapInstaller is active, self.device returns cpu because
        # parameters are swapped to GPU only during forward.  Use the explicitly
        # passed device (the intended execution device) when available.
        target_device = device if device is not None else self.device
        ids = ids.to(target_device)
        mask = mask.to(target_device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        pretrained_path=None,
        z_dim=48,
        vae_type="Wan2.2_VAE",
        wan_models_dir=None,
    ):
        super().__init__()
        if vae_type != "Wan2.2_VAE":
            raise ValueError(f"Unsupported vae_type={vae_type!r}; only 'Wan2.2_VAE' is supported.")
        from abot_world.iris.wan.modules.vae2_2 import _video_vae
        self.mean = torch.tensor([
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
                -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
                -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
                -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
                -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
                0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
        ], dtype=torch.float32)

        self.std = torch.tensor([
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ], dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]
        self.upsampling_factor = 16

        z_dim = 48
        self.z_dim = z_dim
        self.model = _video_vae(pretrained_path=pretrained_path,
            z_dim=z_dim,).eval().requires_grad_(False)

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype, device=device)
        return noise
    
    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False, return_in_cpu: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            if return_in_cpu:
                decoded = decoded.cpu()
            output.append(decoded)
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class TAEW2_2VAEWrapper(torch.nn.Module):
    """
    VAE wrapper using TAEHV (TAEW2.2) for faster decoding.
    Requires: pip install taehv (or install from https://github.com/madebyollin/taehv)
    Checkpoint: taew2_2.pth (download from taehv releases)
    """
    def __init__(self, checkpoint_path: str = "taew2_2.pth", dtype=torch.float16):
        super().__init__()
        try:
            from abot_world.iris.wan.modules.taehv import TAEHV, StreamingTAEHV
        except ImportError as e:
            raise ImportError(
                "taehv is required for TAEW2.2 VAE. Install with: pip install taehv"
            ) from e
        self.taehv = TAEHV(checkpoint_path).to(dtype).eval().requires_grad_(False)
        self.taehv = StreamingTAEHV(self.taehv)
        self.dtype = dtype
        # For compatibility with pipeline.vae.model.clear_cache()
        self.model = _TAEW2_2ModelRef(self)

    def warmup_first_frame(self, first_frame_latent: torch.Tensor):
        """Warm up the streaming decoder's MemBlock memory with the first-frame latent.

        The TAEW2.2 MemBlocks use zero-initialized past context for the first frame,
        causing blur.  By feeding the first-frame latent as a warmup pass (output
        discarded), subsequent decodes benefit from real temporal context.

        Args:
            first_frame_latent: [B, 1, C, H, W] latent of the first frame.
        """
        if first_frame_latent is None:
            return
        # Reset decoder state, then feed first frame as warmup
        self.taehv.reset()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            # Feed the first-frame latent to populate MemBlock memory;
            # the output (startup frames) is discarded.
            _ = self.taehv.decode(first_frame_latent)

    def decode_to_pixel(
        self,
        latent: torch.Tensor,
        use_cache: bool = False,
        return_in_cpu: bool = False
    ) -> torch.Tensor:
        # latent: [B, F, C, H, W] = [B, T, C, H, W] (same as TAEHV's NTCHW)
        # use_cache=True -> parallel=False for lower memory (streaming)
        parallel = not use_cache
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # out = self.taehv.decode_video(
            #     latent, parallel=parallel, show_progress_bar=False
            # )
            out = self.taehv.decode(latent)
        # TAEHV returns [0, 1], convert to [-1, 1] to match WanVAEWrapper
        out = out.mul(2).sub(1).clamp(-1, 1).float()
        if return_in_cpu:
            out = out.cpu()
        return out


class _TAEW2_2ModelRef:
    """Dummy ref for clear_cache compatibility; delegates to StreamingTAEHV.reset()."""

    def __init__(self, parent):
        self._parent = parent

    def clear_cache(self):
        self._parent.taehv.reset()


class MGLightVAEWrapper(torch.nn.Module):
    """VAE wrapper using MG-LightVAE (pruned Wan2.2 VAE) for faster decoding.

    Wraps the ``Wan2_2_VAE`` class which supports different pruning rates.
    The encoder uses the full (unpruned) Wan2.2 VAE teacher, while the decoder
    uses the pruned student model.

    Args:
        vae_pth: Path to the pruned LightVAE checkpoint (student decoder).
        lightvae_pruning_rate: Pruning rate for the decoder (e.g. 0.5, 0.75).
        lightvae_encoder_vae_pth: Path to the full Wan2.2 VAE checkpoint
            (teacher encoder). Required for mg_lightvae.
        dtype: Data type for the VAE model.
        device: Device to load the VAE on.
    """

    def __init__(
        self,
        vae_pth: str,
        lightvae_pruning_rate: float = 0.75,
        lightvae_encoder_vae_pth: str | None = None,
        dtype=torch.float,
        device="cpu",
    ):
        super().__init__()
        from abot_world.iris.wan.modules.vae2_2 import Wan2_2_VAE

        self._vae = Wan2_2_VAE(
            z_dim=48,
            c_dim=160,
            vae_pth=vae_pth,
            dtype=dtype,
            device=device,
            vae_type="mg_lightvae",
            lightvae_pruning_rate=lightvae_pruning_rate,
            lightvae_encoder_vae_pth=lightvae_encoder_vae_pth,
        )
        # Register model (pruned decoder) and encoder_model (teacher encoder)
        # as submodules so that .to(), .eval(), .requires_grad_() propagate.
        self.model = self._vae.model
        if self._vae.encoder_model is not None:
            self.encoder_model = self._vae.encoder_model
        else:
            self.encoder_model = None

        self.z_dim = 48
        self.upsampling_factor = 16
        self.mean = self._vae.scale[0]
        self.std = 1.0 / self._vae.scale[1]

        # Initialize streaming cache attributes (_feat_map, _conv_idx, etc.)
        # so cached_decode() can be called before any explicit clear_cache().
        self.model.clear_cache()
        if self.encoder_model is not None:
            self.encoder_model.clear_cache()

    def generate_noise(self, shape, seed=None, rand_device="cpu",
                        rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype, device=device)
        return noise

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        encode_model = self.encoder_model if self.encoder_model is not None else self.model
        output = [
            encode_model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [B, C, F, H, W] to [B, F, C, H, W]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False,
                        return_in_cpu: bool = False) -> torch.Tensor:
        # from [B, F, C, H, W] to [B, C, F, H, W]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            if return_in_cpu:
                decoded = decoded.cpu()
            output.append(decoded)
        output = torch.stack(output, dim=0)
        # from [B, C, F, H, W] to [B, F, C, H, W]
        output = output.permute(0, 2, 1, 3, 4)
        return output


def create_vae_from_config(config) -> Optional[torch.nn.Module]:
    """Create a VAE wrapper based on the unified ``vae_type`` config field.

    Supported vae_type values:
        - "wan2.2":         Standard Wan2.2 VAE (returns None, pipeline creates WanVAEWrapper)
        - "taew2_2":        TAEW2.2/taehv fast streaming decoder
        - "mg_lightvae":    MG-LightVAE pruned decoder (pruning rate 0.5)
        - "mg_lightvae_v2": MG-LightVAE v2 pruned decoder (pruning rate 0.75)

    If ``vae_type`` is not set, defaults to "taew2_2".

    Returns:
        A VAE wrapper instance, or None for wan2.2 (let pipeline create
        WanVAEWrapper from vae_kwargs).
    """
    vae_type = getattr(config, "vae_type", None)

    if vae_type is None:
        vae_type = "taew2_2"
    else:
        vae_type = str(vae_type).strip().lower()

    if vae_type == "wan2.2":
        return None  # pipeline creates WanVAEWrapper(**vae_kwargs) internally

    if vae_type == "fantasydecoder":
        from abot_world.iris.fantasydecoder import FantasyDecoder
        ckpt = os.environ.get("FANTASYDECODER_CHECKPOINT") or getattr(config, "fantasydecoder_checkpoint", None)
        if not ckpt:
            raise ValueError("Set fantasydecoder_checkpoint or FANTASYDECODER_CHECKPOINT")
        return FantasyDecoder(
            checkpoint_path=_resolve_wan_path(ckpt),
            latent_chunk_size=getattr(config, "fantasydecoder_chunk_size", None),
            cache_window_frames=getattr(config, "fantasydecoder_cache_window", None),
            latent_chunk_causal=getattr(config, "fantasydecoder_chunk_causal", None),
        ).eval()

    if vae_type == "taew2_2":
        ckpt = os.environ.get("TAEW2_2_CHECKPOINT") or getattr(
            config, "taew2_2_checkpoint", "taew2_2.pth"
        )
        return TAEW2_2VAEWrapper(checkpoint_path=ckpt).eval()

    if vae_type in ("mg_lightvae", "mg_lightvae_v2"):
        pruning_map = {"mg_lightvae": 0.5, "mg_lightvae_v2": 0.75}
        # Explicit pruning rate overrides the default mapping
        explicit_rate = getattr(config, "lightvae_pruning_rate", None)
        if explicit_rate is not None:
            pruning_rate = float(explicit_rate)
        else:
            pruning_rate = pruning_map[vae_type]

        # Select checkpoint based on vae_type
        ckpt_map = {
            "mg_lightvae": "lightvae_checkpoint",
            "mg_lightvae_v2": "lightvae_v2_checkpoint",
        }
        vae_ckpt = getattr(config, ckpt_map[vae_type], None)
        if vae_ckpt is None:
            raise ValueError(
                f"vae_type={vae_type!r} requires '{ckpt_map[vae_type]}' config field "
                f"(path to MG-LightVAE .pth file)."
            )

        # Encoder checkpoint: explicit config, or fall back to vae_kwargs.pretrained_path
        encoder_ckpt = getattr(config, "lightvae_encoder_checkpoint", None)
        if encoder_ckpt is None:
            vae_kwargs = getattr(config, "vae_kwargs", {}) or {}
            if isinstance(vae_kwargs, dict):
                encoder_ckpt = vae_kwargs.get("pretrained_path")
            else:
                encoder_ckpt = getattr(vae_kwargs, "pretrained_path", None)
        if encoder_ckpt is None:
            raise ValueError(
                f"vae_type={vae_type!r} requires 'lightvae_encoder_checkpoint' config field "
                f"(path to full Wan2.2_VAE.pth for teacher encoder), "
                f"or 'vae_kwargs.pretrained_path' must be set."
            )

        return MGLightVAEWrapper(
            vae_pth=vae_ckpt,
            lightvae_pruning_rate=pruning_rate,
            lightvae_encoder_vae_pth=encoder_ckpt,
        )

    raise ValueError(
        f"Unsupported vae_type={vae_type!r}. "
        f"Choose from: wan2.2, taew2_2, mg_lightvae, mg_lightvae_v2."
    )


class WanDiffusionWrapper(torch.nn.Module):
    @staticmethod
    def _materialize_meta_tensors(module: torch.nn.Module, device: torch.device = torch.device("cpu")):
        materialized_names = []

        def _materialize_recursive(mod: torch.nn.Module, prefix: str = ""):
            for name, param in list(mod.named_parameters(recurse=False)):
                if getattr(param, "is_meta", False):
                    new_param = torch.nn.Parameter(
                        torch.empty(tuple(param.shape), dtype=param.dtype, device=device),
                        requires_grad=param.requires_grad,
                    )
                    setattr(mod, name, new_param)
                    materialized_names.append(prefix + name)

            for name, buf in list(mod.named_buffers(recurse=False)):
                if getattr(buf, "is_meta", False):
                    setattr(mod, name, torch.empty(tuple(buf.shape), dtype=buf.dtype, device=device))
                    materialized_names.append(prefix + name)

            for child_name, child in mod.named_children():
                _materialize_recursive(child, prefix + child_name + ".")

        _materialize_recursive(module)
        return materialized_names

    @staticmethod
    def _normalize_model_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        key_prefixes = (
            "generator.model._fsdp_wrapped_module.",
            "generator.model.",
            "model._fsdp_wrapped_module.",
            "model.",
            "_fsdp_wrapped_module.",
            "module.",
        )
        keys = list(state_dict.keys())
        for prefix in key_prefixes:
            if keys and all(k.startswith(prefix) for k in keys):
                return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict

    def _load_model_safetensors(self, model_safetensors_path: str) -> None:
        if model_safetensors_path.startswith("oss://"):
            raise ValueError(
                "model_safetensors_path must be a local mounted path on AI-Hub, "
                f"got {model_safetensors_path}"
            )

        model_safetensors_path = _resolve_wan_path(model_safetensors_path)
        print(f"[WanDiffusionWrapper] Loading model safetensors from {model_safetensors_path}")
        state_dict = load_safetensors_file(model_safetensors_path, device="cpu")
        state_dict = self._normalize_model_state_dict_keys(state_dict)

        model_keys = set(self.model.state_dict().keys())
        matched_keys = model_keys.intersection(state_dict.keys())
        if not matched_keys:
            sample_keys = list(state_dict.keys())[:10]
            raise ValueError(
                "No safetensors keys matched the Wan model state_dict. "
                f"First loaded keys: {sample_keys}"
            )

        match_ratio = len(matched_keys) / max(1, len(state_dict))
        if match_ratio < 0.5:
            sample_unexpected = [k for k in state_dict.keys() if k not in model_keys][:10]
            raise ValueError(
                f"Only {len(matched_keys)}/{len(state_dict)} safetensors keys match the Wan model "
                f"state_dict after prefix normalization. Sample unexpected keys: {sample_unexpected}"
            )

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"[WanDiffusionWrapper] model_safetensors missing {len(missing)} keys "
                f"(showing first 20): {missing[:20]}"
            )
        if unexpected:
            print(
                f"[WanDiffusionWrapper] model_safetensors unexpected {len(unexpected)} keys "
                f"(showing first 20): {unexpected[:20]}"
            )
        print(
            f"[WanDiffusionWrapper] Loaded model safetensors with {len(matched_keys)} "
            f"matched keys from {model_safetensors_path}"
        )

    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            subfolder=None,
            model_type='t2v',
            num_frame_per_block=3,
            model_safetensors_path: Optional[str] = None,
            inference_checkpoint: Optional[str] = None,
            inference_use_ema: bool = False,
            **model_init_kwargs,
    ):
        super().__init__()

        self.model_type = model_type
        use_relative_rope = bool(model_init_kwargs.pop("use_relative_rope", False))
        if inference_checkpoint:
            if model_safetensors_path:
                raise ValueError("Use either inference_checkpoint or model_safetensors_path")
            from accelerate import init_empty_weights
            model_class = CausalWanModel if is_causal else WanModel
            options = dict(model_init_kwargs, model_type=model_type)
            if is_causal:
                options.update(use_relative_rope=use_relative_rope,
                               local_attn_size=local_attn_size, sink_size=sink_size,
                               num_frame_per_block=num_frame_per_block)
            with init_empty_weights():
                self.model = model_class.from_config(model_class.load_config(model_name), **options)
            weights = torch.load(inference_checkpoint, map_location="cpu", mmap=True, weights_only=True)
            key = "generator_ema" if inference_use_ema else "generator"
            self.load_state_dict(weights[key], strict=True, assign=True)
            del weights
        elif is_causal:
            model_init_kwargs["use_relative_rope"] = use_relative_rope
            self.model = CausalWanModel.from_pretrained(
                model_name, local_attn_size=local_attn_size, sink_size=sink_size, model_type=model_type, num_frame_per_block=num_frame_per_block,
                **model_init_kwargs)
        else:
            if use_relative_rope:
                print("[WanDiffusionWrapper] use_relative_rope is ignored for non-causal WanModel.")
            self.model = WanModel.from_pretrained(model_name, model_type=model_type, **model_init_kwargs)
        materialized = self._materialize_meta_tensors(self.model, device=torch.device("cpu"))
        if materialized:
            print(f"[WanDiffusionWrapper] Materialized {len(materialized)} meta tensors on CPU.")
        if model_safetensors_path:
            self._load_model_safetensors(model_safetensors_path)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = None  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    @staticmethod
    def _history_x_to_model_format(history_x):
        if history_x is None:
            return None

        if torch.is_tensor(history_x):
            if history_x.ndim != 5:
                raise ValueError(
                    f"history_x must be [B,F,C,H,W] when passed as a tensor, got {history_x.shape}"
                )
            return [u.permute(1, 0, 2, 3).contiguous() for u in history_x]

        return history_x

    @staticmethod
    def _history_condition_to_model_format(value):
        if value is None:
            return None

        if torch.is_tensor(value):
            if value.ndim < 3:
                return value
            return [u.contiguous() for u in value]

        return value

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,

        clean_x: Optional[torch.Tensor] = None, # TF
        aug_t: Optional[torch.Tensor] = None, # for TF clean GT, if it's also noisy and needs denoising by the model, aug_t is its timestep

        cache_start: Optional[int] = None,
        updating_cache: Optional[bool] = False,
        replace_first_timestep_and_noise_latents: Optional[bool] = False,
        history_x: Optional[torch.Tensor] = None,
        history_y: Optional[torch.Tensor] = None,
        history_act_context: Optional[torch.Tensor] = None,
        history_y_action: Optional[torch.Tensor] = None,
        noisy_start_frame: int = 0,

    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        act_context = conditional_dict.get("act_context", None)
        act_context_scale = conditional_dict.get("act_context_scale", 1.0)
        clip_fea = conditional_dict.get("clip_fea", None)
        y = conditional_dict.get("y", None)
        y_action = conditional_dict.get("y_action", None)
        ref_latents = conditional_dict.get("ref_latents", None)
        ref_mask = conditional_dict.get("ref_mask", None)
        # first_frame_latents = conditional_dict.get("first_frame_latents", None)

        raw_timestep = timestep
        b, f, c, h, w = noisy_image_or_video.shape

        if replace_first_timestep_and_noise_latents:
            # Wan2.2 5B uses the first latent frame as a clean condition. Keep
            # per-frame timesteps for score models so only frame 0 is forced to t=0.
            if raw_timestep.dim() == 2:
                input_timestep = raw_timestep.clone()
                input_timestep[:, 0] = 0
            elif raw_timestep.dim() == 1 and raw_timestep.shape[0] == f:
                input_timestep = raw_timestep.unsqueeze(0).repeat(b, 1)
                input_timestep[:, 0] = 0
            elif raw_timestep.dim() == 1 and raw_timestep.shape[0] == b:
                input_timestep = raw_timestep[:, None].repeat(1, f)
                input_timestep[:, 0] = 0
            else:
                input_timestep = raw_timestep.reshape(-1)[0].view(1, 1).repeat(b, f)
                input_timestep[:, 0] = 0
        elif self.uniform_timestep:
            # [B, F] -> [B] for legacy non-causal uniform score models.
            input_timestep = raw_timestep[:, 0]
        else:
            input_timestep = raw_timestep

        if history_x is not None:
            history_kwargs = {
                "history_x": self._history_x_to_model_format(history_x),
                "noisy_start_frame": int(noisy_start_frame),
            }
            history_y = self._history_condition_to_model_format(history_y)
            history_act_context = self._history_condition_to_model_format(history_act_context)
            history_y_action = self._history_condition_to_model_format(history_y_action)

            if history_y is not None:
                history_kwargs["history_y"] = history_y
            if history_act_context is not None:
                history_kwargs["history_act_context"] = history_act_context
            if history_y_action is not None:
                history_kwargs["history_y_action"] = history_y_action

            model_out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=None,
                crossattn_cache=crossattn_cache,
                current_start=0 if current_start is None else current_start,
                cache_start=0 if cache_start is None else cache_start,
                act_context=act_context,
                y_action=y_action,
                act_context_scale=act_context_scale,
                clip_fea=clip_fea,
                y=y,
                ref_latents=ref_latents,
                ref_mask=ref_mask,
                **history_kwargs,
            )
            if isinstance(model_out, tuple):
                flow_pred = model_out[0]
            else:
                flow_pred = model_out
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        # X0 prediction
        elif kv_cache is not None:
            kwargs = {}
            if updating_cache:
                kwargs["updating_cache"] = updating_cache
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W],
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                act_context=act_context,
                act_context_scale=act_context_scale,
                clip_fea=clip_fea,
                y=y,
                ref_latents=ref_latents,
                ref_mask=ref_mask,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    aug_t=aug_t,
                    act_context=act_context,
                    act_context_scale=act_context_scale,
                    clip_fea=clip_fea,
                    y=y,
                    ref_latents=ref_latents,
                    ref_mask=ref_mask,
                ).permute(0, 2, 1, 3, 4)
            else:
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    act_context=act_context,
                    act_context_scale=act_context_scale,
                    clip_fea=clip_fea,
                    y=y,
                    ref_latents=ref_latents,
                    ref_mask=ref_mask,
                ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
