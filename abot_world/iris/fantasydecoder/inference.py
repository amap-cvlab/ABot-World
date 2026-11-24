"""Wan2.2 normalized latents -> pixels, using the trained FantasyDecoder."""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from .config import FlashDecoderConfig
from .model import FlashDecoder


class _ModelRef:
    """Keep existing pipeline.vae.model.clear_cache() call sites working."""

    def __init__(self, parent):
        self.parent = parent
        self.z_dim = parent.z_dim

    def clear_cache(self):
        self.parent.reset_streaming_state()


class FantasyDecoder(nn.Module):
    """Decoder-only adapter with FantasyEarth's [B,T,C,H,W] interface.

    Input is ALREADY normalized by Wan2.2's encoder / DiT. Do not apply the
    Wan latent mean/std again. Streaming calls must contain only NEW latents.
    Each call is split into groups of at most ``latent_chunk_size``; trailing
    partial groups are decoded immediately, not buffered across calls.
    """

    def __init__(self, checkpoint_path, latent_chunk_size=None,
                 cache_window_frames=None, latent_chunk_causal=None):
        super().__init__()
        path = Path(checkpoint_path).expanduser()
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        model_config = dict(payload["config"]["model"], gradient_checkpointing=False)
        if cache_window_frames is not None:
            model_config["window_frames"] = int(cache_window_frames)
        config = FlashDecoderConfig.from_dict(model_config)
        if config.temporal_layout != "wan" or config.latent_channels != 48:
            raise ValueError("FantasyEarth 5B requires a Wan2.2 (48-channel) checkpoint, not H3/Wan2.1")
        training = payload["config"].get("training", {})
        self.latent_chunk_size = int(latent_chunk_size if latent_chunk_size is not None
                                     else training.get("latent_chunk_size", 1))
        if self.latent_chunk_size < 1:
            raise ValueError("latent_chunk_size must be positive")
        self.latent_chunk_causal = (bool(latent_chunk_causal) if latent_chunk_causal is not None
                                    else bool(training.get("latent_chunk_causal", False)))
        self.decoder = FlashDecoder(config)
        self.decoder.load_state_dict(payload["model"], strict=True)
        self.decoder.eval().requires_grad_(False)
        self.z_dim = config.latent_channels
        self.upsampling_factor = config.spatial_factor
        self.checkpoint_path = str(path)
        self.checkpoint_step = payload.get("step")
        self.model = _ModelRef(self)
        self.reset_streaming_state()
        print(f"[FantasyDecoder] checkpoint={path} step={self.checkpoint_step} "
              f"chunk={self.latent_chunk_size} cache={config.window_frames} "
              f"causal_within_chunk={self.latent_chunk_causal} backend=PyTorch-SDPA", flush=True)

    def reset_streaming_state(self):
        self._state = None
        self._stream_shape = None
        self._pending_latents = None

    @torch.inference_mode()
    def decode_stream(self, latent=None, *, final=False, return_in_cpu=False):
        """Decode complete groups across calls; only ``final`` permits a tail.

        Returns None while fewer than one group is available. Reset discards
        an unfinished tail. Do not mix this entry point with cached immediate
        decode_to_pixel calls within a stream.
        """
        pending = self._pending_latents
        if latent is not None:
            if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != self.z_dim or latent.shape[1] == 0:
                raise ValueError("Expected nonempty normalized Wan2.2 latents [1,T,48,H,W]")
            param = next(self.decoder.parameters())
            if latent.device != param.device:
                raise ValueError("Move FantasyDecoder to the same device as the input before decoding")
            shape = (latent.shape[0], latent.shape[3], latent.shape[4])
            if ((self._stream_shape is not None and shape != self._stream_shape)
                    or (pending is not None and shape != (pending.shape[0], pending.shape[3], pending.shape[4]))):
                raise ValueError("Stream resolution changed; call model.clear_cache() first")
            latent = torch.cat((pending, latent), dim=1) if pending is not None else latent
        else:
            latent = pending
        if latent is None:
            return None
        count = latent.shape[1] if final else latent.shape[1] // self.latent_chunk_size * self.latent_chunk_size
        pixels = (self.decode_to_pixel(latent[:, :count], use_cache=True, return_in_cpu=return_in_cpu)
                  if count else None)
        # A small view would retain the whole input allocation between calls.
        self._pending_latents = latent[:, count:].detach().clone() if count < latent.shape[1] else None
        return pixels

    def _apply(self, fn, recurse=True):
        # KV tensors are not module buffers. Never retain caches after a
        # device/dtype change (e.g. UI offload or moving a reused pipeline).
        self.reset_streaming_state()
        return super()._apply(fn, recurse=recurse)

    def encode_to_latent(self, pixel):
        raise NotImplementedError("Use the pipeline's separate Wan2.2 encoder for reference images")

    @torch.inference_mode()
    def decode_to_pixel(self, latent, use_cache=False, return_in_cpu=False):
        if latent.ndim != 5 or latent.shape[2] != self.z_dim or latent.shape[1] == 0:
            raise ValueError("Expected nonempty normalized Wan2.2 latents [B,T,48,H,W]")
        if use_cache and latent.shape[0] != 1:
            raise ValueError("Cached decoding requires batch size 1")
        param = next(self.decoder.parameters())
        if latent.device != param.device:
            raise ValueError("Move FantasyDecoder to the same device as the input before decoding")
        shape = (latent.shape[0], latent.shape[3], latent.shape[4])
        if use_cache and self._stream_shape is not None and shape != self._stream_shape:
            raise ValueError("Stream resolution changed; call model.clear_cache() first")
        z = latent.permute(0, 2, 1, 3, 4).to(dtype=param.dtype)
        context = (torch.autocast("cuda", dtype=param.dtype)
                   if param.device.type == "cuda" and param.dtype in (torch.float16, torch.bfloat16)
                   else nullcontext())
        with context:
            if not use_cache:
                self.reset_streaming_state()
                pixels = self.decoder(z, latent_chunk_size=self.latent_chunk_size,
                                      latent_chunk_causal=self.latent_chunk_causal).permute(0, 2, 1, 3, 4)
            else:
                state = self._state if self._state is not None else self.decoder.empty_state()
                outputs = []
                for start in range(0, z.shape[2], self.latent_chunk_size):
                    pixels, state = self.decoder.stream_chunk(
                        z[:, :, start:start + self.latent_chunk_size], state,
                        causal_within_chunk=self.latent_chunk_causal)
                    outputs.append(pixels)
                pixels = torch.cat(outputs, dim=1)
                self._state, self._stream_shape = state, shape
        pixels = pixels.float().clamp(-1, 1)
        return pixels.cpu() if return_in_cpu else pixels
