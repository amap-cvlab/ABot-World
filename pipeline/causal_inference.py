from typing import List, Optional
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as Functional
from PIL import Image, ImageOps
from einops import repeat

from utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
    model_kwargs_with_relative_rope,
)

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, \
    move_model_to_device_with_memory_preservation
import tqdm

# Supported image extensions for reference image loading (case-insensitive)
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")


def _resolve_ref_image_path(directory: Path, name: str) -> Optional[Path]:
    """Resolve a reference image name to an actual file with any supported extension."""
    for ext in SUPPORTED_IMAGE_EXTS:
        for variant in (ext, ext.upper()):
            candidate = directory / f"{name}{variant}"
            if candidate.is_file():
                return candidate
    return None


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae=True
    ):
        super().__init__()
        # Step 1: Initialize all models
        model_kwargs = model_kwargs_with_relative_rope(args)
        model_kwargs["local_attn_size"] = model_kwargs.get("local_attn_size", getattr(args, "local_attn_size", -1))
        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=True, model_type=args.model_type,
        ) if generator is None else generator
        self.text_encoder = WanTextEncoder(tokenizer_path=args.text_encoder_kwargs.tokenizer_path,
                                           encoder_pth_path=args.text_encoder_kwargs.encoder_pth_path) if text_encoder is None else text_encoder
        if vae is not None and getattr(args, "model_type", "t2v") in ["ci2v", "ti2v"]:
            # self.encoder = WanVAEWrapper()
            self.encoder = WanVAEWrapper(**getattr(args, "vae_kwargs", {}))
        else:
            self.encoder = None
        # self.vae = WanVAEWrapper() if vae is None else vae
        self.vae = WanVAEWrapper(**getattr(args, "vae_kwargs", {})) if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = (
            args.image_or_video_shape[3] * args.image_or_video_shape[4]
        ) // (
            self.generator.model.patch_size[1] * self.generator.model.patch_size[2]
        )

        self.kv_cache1 = None
        self.args = args
        self.device = device
        self.torch_dtype = torch.bfloat16
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.pyramid_sample_ratio = getattr(args, "pyramid_sample_ratio", None)

        self.conditional_dict = None
        self.crossattn_cache = None

        # Streaming state
        self.current_start_frame = 0
        self.num_input_frames = 0
        self._stream_block_diffusion_times: List[float] = []
        self._stream_block_decode_times: List[float] = []

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _clear_generation_caches(self):
        self.kv_cache1 = None
        self.crossattn_cache = None
        model = getattr(self.generator, "model", None)
        if model is not None and hasattr(model, "clear_cache"):
            model.clear_cache()

    def _decode_output_to_video(self, output: torch.Tensor) -> torch.Tensor:
        self._clear_generation_caches()
        offload = bool(getattr(self.args, "offload_generator_for_decode", False))
        generator_device = None
        if offload:
            try:
                generator_device = next(self.generator.parameters()).device
            except StopIteration:
                generator_device = None
            if generator_device is not None and generator_device.type == "cuda":
                self.generator.to("cpu")
                torch.cuda.empty_cache()

        try:
            return self.vae.decode_to_pixel(output, use_cache=False)
        finally:
            if (
                offload
                and generator_device is not None
                and generator_device.type == "cuda"
                and bool(getattr(self.args, "restore_generator_after_decode", False))
            ):
                self.generator.to(generator_device)

    def get_condition_split(self, conditional_dict, start, end):
        # print(f"Getting condition split for frames {start} to {end}",conditional_dict.keys())
        device = conditional_dict["prompt_embeds"].device
        act_context = conditional_dict.get("act_context")
        y = conditional_dict.get("y")
        clip_fea = conditional_dict.get("clip_fea")
        first_frame_latents = conditional_dict.get("first_frame_latents")
        split_conditional_dict = {
            "prompt_embeds": conditional_dict["prompt_embeds"],
            "act_context": act_context[:, :, start:end].to(device) if act_context is not None else None,
            "act_context_scale": conditional_dict.get("act_context_scale", 0.0),
            "y": y[:, :, start:end].to(device) if y is not None else None,
            "clip_fea": clip_fea,
            "first_frame_latents": first_frame_latents,
            "ref_latents": conditional_dict.get("ref_latents", None),
            "ref_mask": conditional_dict.get("ref_mask", None),
        }
        return split_conditional_dict


    def set_prompts(self, text_prompts: List[str], device: torch.device):
        """
        刷新你的世界：重新算 conditional_dict，并让 crossattn_cache 在下次 generator 调用时重建。
        """
        self.conditional_dict = self.text_encoder(text_prompts=text_prompts, device=device)

        # 强制 cross-attn cache 重新 init（因为 prompt 变了）
        if hasattr(self, "crossattn_cache") and self.crossattn_cache is not None:
            for b in range(self.num_transformer_blocks):
                self.crossattn_cache[b]["is_init"] = False

    def set_ref_latent_mask_from_exists_paths(
        self,
        ref_dir: str,
        device: Optional[torch.device] = None,
    ):
        """Set ref_latents and ref_mask from a directory of pre-generated reference images.

        ``ref_dir`` is expected to contain the 5 output images (head, left,
        right, front, back) in any common image format (.jpg, .jpeg, .png,
        .webp, .bmp, .tiff). The method only performs the VAE encoding step.

        If any expected file is missing, falls back to zero ref_latents with zero
        ref_mask (model ignores them).
        Should be called after set_prompts().
        """
        if self.conditional_dict is None:
            raise RuntimeError("call set_prompts first")
        dev = device or self.device
        num_slots = getattr(self.args, "ref_num_slots", 5)
        ref_resolution = getattr(self.args, "ref_resolution", 512)
        # Get z_dim and upsampling_factor from encoder (preferred) or vae
        vae_for_shape = self.encoder if self.encoder is not None else self.vae
        z_dim = getattr(vae_for_shape, "z_dim", 48)
        upsampling = getattr(vae_for_shape, "upsampling_factor", 16)
        ref_h = ref_w = ref_resolution // upsampling

        ref_names = ["head", "left", "right", "front", "back"]
        ref_dir_path = Path(ref_dir)

        # Resolve each reference name to an actual file with any supported extension
        resolved_paths = []
        missing = []
        for name in ref_names[:num_slots]:
            found = _resolve_ref_image_path(ref_dir_path, name)
            if found is not None:
                resolved_paths.append(found)
            else:
                missing.append(name)

        if missing:
            print(f"[set_ref_latent_mask_from_exists_paths] Missing ref images {missing} "
                  f"in {ref_dir}, falling back to zero ref_latents.")
            ref_latents = torch.zeros(
                [1, num_slots, z_dim, 1, ref_h, ref_w],
                dtype=torch.bfloat16, device=dev,
            )
            ref_mask = torch.zeros([1, num_slots], dtype=torch.float32, device=dev)
            source = "zeros_missing"
        else:
            vae = self.encoder if self.encoder is not None else self.vae
            vae_device = next(vae.parameters()).device
            print(
                f"[set_ref_latent_mask_from_exists_paths] encoding {len(resolved_paths)} refs "
                f"on device={vae_device} (vae={type(vae).__name__})",
                flush=True,
            )
            if vae_device.type == "cpu":
                print(
                    "[set_ref_latent_mask_from_exists_paths] WARNING: encoder is on CPU; "
                    "Wan2.2 VAE encode will be extremely slow. "
                    "Ensure pipeline.encoder is moved to CUDA in pipeline_loader.",
                    flush=True,
                )

            latents_list = []
            for i, img_path in enumerate(resolved_paths):
                print(
                    f"[set_ref_latent_mask_from_exists_paths] encode [{i+1}/{len(resolved_paths)}] "
                    f"{img_path.name} ...",
                    flush=True,
                )
                img = Image.open(img_path).convert("RGB")
                img = img.resize((ref_resolution, ref_resolution))
                image = self.preprocess_image(img, device=dev, torch_dtype=torch.bfloat16)
                ref_pixel = image.unsqueeze(2)  # [1, C, 1, H, W]
                with torch.no_grad():
                    latent = vae.encode_to_latent(ref_pixel.to(vae_device))
                latent_slot = latent[0, 0].unsqueeze(1).to(device=dev, dtype=torch.bfloat16)
                latents_list.append(latent_slot)

            ref_latents = torch.stack(latents_list, dim=0).unsqueeze(0)  # [1, K, C_lat, 1, H_lat, W_lat]
            ref_mask = torch.ones([1, num_slots], dtype=torch.float32, device=dev)
            source = "exists_paths"

        self.conditional_dict["ref_latents"] = ref_latents
        self.conditional_dict["ref_mask"] = ref_mask
        print(f"[set_ref_latent_mask_from_exists_paths] ref_latents={ref_latents.shape}, "
              f"ref_mask={ref_mask.shape}, "
              f"source={source}")

    def preprocess_image(self, image, torch_dtype=torch.bfloat16, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype, device=device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    @torch.no_grad()
    def set_first_frame_latent(
        self,
        ref_image_path: str,
        height: int = 480,
        width: int = 832,
        device: Optional[torch.device] = None,
    ):
        """
        设置首帧 clean latent：从路径加载图片，通过 vae 编码得到 latent，
        存入 self.conditional_dict["first_frame_latents"]。
        在 generate_next_block 的第一个 block 中会用它替换噪声的首帧。
        """
        if self.conditional_dict is None:
            raise RuntimeError("call set_prompts first")
        dev = device or self.device
        if self.encoder is not None:
            vae = self.encoder
        else:
            vae = self.vae

        if not ref_image_path:
            self.conditional_dict["first_frame_latents"] = None
        else:
            input_image = Image.open(ref_image_path).convert("RGB")
            # Preserve the source aspect ratio: scale to cover the target canvas
            # then center-crop, instead of non-uniform stretching to (width, height).
            input_image = ImageOps.fit(
                input_image,
                (width, height),
                method=Image.LANCZOS,
                centering=(0.5, 0.5),
            )
            image = self.preprocess_image(input_image, device=dev, torch_dtype=torch.bfloat16)
            ref_pixel = image.unsqueeze(2)  # [B, C, 1, H, W]
            # encode_to_latent returns [B, 1, C_lat, H_lat, W_lat]
            first_frame_latents = vae.encode_to_latent(ref_pixel.to(next(vae.parameters()).device))
            self.conditional_dict["first_frame_latents"] = first_frame_latents.to(device=dev, dtype=torch.bfloat16)
            print(f"set_first_frame_latent: shape={first_frame_latents.shape}")

    def set_act(self, keys_dict, height: int = 480, width: int = 832, num_frames: int = 81, device: Optional[torch.device] = None):
        """
        设置 act_context [B, 8]
        device: 指定设备，默认 self.device
        """
        if self.conditional_dict is None:
            raise RuntimeError("call set_prompts first")
        key_order = ['W', 'A', 'S', 'D', 'I', 'J', 'K', 'L']
        action_list = [1 if keys_dict.get(k, False) else 0 for k in key_order]
        action = torch.tensor(action_list, dtype=torch.float32, device=device).unsqueeze(0)
        control_action_latents = action[:, None, None, :].repeat(1, height, width, 1).permute([3, 0, 1, 2]).unsqueeze(0).to(device=device, dtype=torch.bfloat16) #f,h,w,8 --> 1,8,f,h,w, ([1, 8, 1, 480, 832])
        # repeat control_action_latents to num_frames
        control_action_latents = control_action_latents.repeat_interleave(4, dim=1)  # [1, 8*4, 1, 480, 832]
        control_action_latents = control_action_latents.repeat(1, 1, num_frames, 1, 1)  # [1, 8*4, F, 480, 832]
        self.conditional_dict["act_context"] = control_action_latents
        print(f'action_list, {action_list}')


    def reset_stream(self, batch_size: int, dtype, device, initial_latent=None):
        ref_cache_token_len = self._ref_cache_token_len(getattr(self, 'conditional_dict', None))
        if self._kv_cache_needs_reinit(ref_cache_token_len):
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=dtype, device=device,
                extra_tokens=ref_cache_token_len,
            )
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
        else:
            # reset cross-attn cache flags
            for b in range(self.num_transformer_blocks):
                self.crossattn_cache[b]["is_init"] = False
            # reset kv cache indices
            for b in range(len(self.kv_cache1)):
                self.kv_cache1[b]["global_end_index"].fill_(0)
                self.kv_cache1[b]["local_end_index"].fill_(0)

        self.current_start_frame = 0
        self.num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        self._stream_block_diffusion_times = []
        self._stream_block_decode_times = []


    @torch.no_grad()
    def generate_next_block(self, noise_block: torch.Tensor):
        """
        noise_block: [B, F, C, H, W]，其中 F=1 或 num_frame_per_block
        return denoised_pred: [B, F, C, H, W] on GPU
        """
        t0 = time.perf_counter()
        B, F, C, H, W = noise_block.shape
        noisy_input = noise_block

        y = self.conditional_dict.get("y", None)
        if y is not None:
            y = y.clone()

        action_context = self.conditional_dict.get("act_context", None)
        if action_context is not None:
            action_context = action_context.clone()
            _, Ca, _, Ha, Wa = action_context.shape

        first_frame_latents = self.conditional_dict.get("first_frame_latents", None)
        replace_first = (self.current_start_frame - self.num_input_frames == 0) and (first_frame_latents is not None)
        if replace_first:
            print(f"replace the first timestep with the clean context. latent shape: {noisy_input.shape}, first_frame_latents shape: {first_frame_latents.shape}")
            noisy_input[:, 0:1] = first_frame_latents

        #Pyramid denoising
        if self.pyramid_sample_ratio is not None:
            assert len(self.pyramid_sample_ratio) == len(self.denoising_step_list)
            noisy_input = noisy_input.reshape(B, F*C, H, W)
            noisy_input = Functional.interpolate(noisy_input, scale_factor=self.pyramid_sample_ratio[0], mode='nearest')
            noisy_input = noisy_input.reshape(B, F, C, int(H*self.pyramid_sample_ratio[0]), int(W*self.pyramid_sample_ratio[0]))

            if y is not None:
                cur_y = y.reshape(B, C*F, H, W)
                cur_y = Functional.interpolate(cur_y, scale_factor=self.pyramid_sample_ratio[0], mode='nearest')
                cur_y = cur_y.reshape(B, C, F, int(H*self.pyramid_sample_ratio[0]), int(W*self.pyramid_sample_ratio[0]))
                self.conditional_dict["y"] = cur_y.clone()

            if action_context is not None:
                cur_action_context = action_context.reshape(B, Ca*F, Ha, Wa)
                cur_action_context = Functional.interpolate(cur_action_context, scale_factor=self.pyramid_sample_ratio[0], mode='nearest')
                cur_action_context = cur_action_context.reshape(B, Ca, F, int(Ha*self.pyramid_sample_ratio[0]), int(Wa*self.pyramid_sample_ratio[0]))
                self.conditional_dict["act_context"] = cur_action_context.clone()


        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones([B, F], device=noise_block.device, dtype=torch.int64) * current_timestep
            if replace_first:
                timestep[:, 0] = 0

            if self.pyramid_sample_ratio is not None:
                _, _, _, H_s, W_s = noisy_input.shape

            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.current_start_frame * self.frame_seq_length,
                replace_first_timestep_and_noise_latents=replace_first,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                if self.pyramid_sample_ratio is not None:
                    denoised_pred = denoised_pred.reshape(B, F*C, H_s, W_s)
                    denoised_pred = Functional.interpolate(denoised_pred, scale_factor=self.pyramid_sample_ratio[index+1], mode='nearest')
                    denoised_pred = denoised_pred.reshape(B, F, C, int(H_s*self.pyramid_sample_ratio[index+1]), int(W_s*self.pyramid_sample_ratio[index+1]))

                    if y is not None:
                        cur_y = y.reshape(B, C*F, H, W)
                        scale = math.prod(self.pyramid_sample_ratio[:index+2])
                        cur_y = Functional.interpolate(cur_y, scale_factor=scale, mode='nearest')
                        cur_y = cur_y.reshape(B, C, F, int(H*scale), int(W*scale))
                        self.conditional_dict["y"] = cur_y

                    if action_context is not None:
                        cur_action_context = action_context.reshape(B, Ca*F, Ha, Wa)
                        scale = math.prod(self.pyramid_sample_ratio[:index+2])
                        cur_action_context = Functional.interpolate(cur_action_context, scale_factor=scale, mode='nearest')
                        cur_action_context = cur_action_context.reshape(B, Ca, F, int(Ha*scale), int(Wa*scale))
                        self.conditional_dict["act_context"] = cur_action_context.clone()

                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([B * F], device=noise_block.device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                noisy_input = denoised_pred

            if replace_first:
                noisy_input[:, 0:1] = first_frame_latents

        # 用干净 context 更新 KV cache
        context_timestep = torch.ones([B, F], device=noise_block.device, dtype=torch.int64) * self.args.context_noise
        if replace_first:
            context_timestep = context_timestep.clone()
            context_timestep[:, 0] = 0
            noisy_input[:, 0:1] = first_frame_latents

        self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=self.current_start_frame * self.frame_seq_length,
            replace_first_timestep_and_noise_latents=replace_first,
        )

        self.current_start_frame += F

        torch.cuda.synchronize()
        block_diffusion_s = time.perf_counter() - t0
        self._stream_block_diffusion_times.append(block_diffusion_s)

        return noisy_input
    
    @torch.no_grad()
    def decode_block_and_write(self, latents_block: torch.Tensor, writer):
        # latents_block: [B,F,C,H,W] on GPU
        t0 = time.perf_counter()
        vid = self.vae.decode_to_pixel(latents_block, use_cache=True, return_in_cpu=True)
        torch.cuda.synchronize()
        block_decode_s = time.perf_counter() - t0
        self._stream_block_decode_times.append(block_decode_s)

        vid = (vid * 0.5 + 0.5).clamp(0, 1)
        # print('decode_block_and_write: ', latents_block.shape, vid.shape) # [1, 3, 16, 60, 104]) torch.Size([1, 12, 3, 480, 832])
        frames = (vid[0].permute(0,2,3,1) * 255.0).clamp(0,255).to(torch.uint8).numpy()
        for f in frames:
            writer.append_data(f)
    
    def inference(
            self,
            noise: torch.Tensor,
            text_prompts: List[str],
            conditional_dict=None,
            initial_latent: Optional[torch.Tensor] = None,
            return_latents: bool = False,
            profile: bool = False,
            low_memory: bool = False,
            rectified_tf=False
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame:
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is not None:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames if self.independent_first_frame else num_frames + num_input_frames
        # conditional_dict = self.text_encoder(
        #     text_prompts=text_prompts
        # )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu,
                                                          preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        ref_cache_token_len = self._ref_cache_token_len(conditional_dict)
        if self._kv_cache_needs_reinit(ref_cache_token_len):
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                extra_tokens=ref_cache_token_len,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["ref_token_len"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                tmp_conditional_dict = self.get_condition_split(conditional_dict, 0, 1)
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=tmp_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                tmp_conditional_dict = self.get_condition_split(conditional_dict, current_start_frame,
                                                                current_start_frame + self.num_frame_per_block)

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=tmp_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm.tqdm(all_num_frames):
            if profile:
                block_start.record()

            if self.independent_first_frame:
                noise_start_frame = current_start_frame
                noise_end_frame = current_start_frame + current_num_frames
            else:
                noise_start_frame = current_start_frame - num_input_frames
                noise_end_frame = current_start_frame + current_num_frames - num_input_frames
            noisy_input = noise[:, noise_start_frame:noise_end_frame]
            condition_start_frame = current_start_frame if self.independent_first_frame else noise_start_frame
            condition_end_frame = (
                current_start_frame + current_num_frames
                if self.independent_first_frame
                else noise_end_frame
            )
            tmp_conditional_dict = self.get_condition_split(
                conditional_dict,
                condition_start_frame,
                condition_end_frame,
            )
            latents = noisy_input
            replace_first_timestep_and_noise_latents = (
                current_start_frame == 0
                and num_input_frames == 0
                and conditional_dict.get("first_frame_latents") is not None
            )
            if replace_first_timestep_and_noise_latents:
                print(f"replace the first timestep with the clean context. latent shape: {noisy_input.shape}, conditional_dict['first_frame_latents'] shape: {conditional_dict['first_frame_latents'].shape}" )
                noisy_input[:, 0:1] = conditional_dict["first_frame_latents"] # replace the first timestep with the clean context

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=tmp_conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        replace_first_timestep_and_noise_latents=replace_first_timestep_and_noise_latents,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=tmp_conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        replace_first_timestep_and_noise_latents=replace_first_timestep_and_noise_latents,
                    )
                    noisy_input = denoised_pred
                if replace_first_timestep_and_noise_latents:
                    noisy_input[:, 0:1] = conditional_dict["first_frame_latents"]
            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = noisy_input

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            if replace_first_timestep_and_noise_latents:
                context_timestep = context_timestep.clone()
                context_timestep[:, 0] = 0
                noisy_input[:, 0:1] = conditional_dict["first_frame_latents"]

            self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=tmp_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                replace_first_timestep_and_noise_latents=replace_first_timestep_and_noise_latents,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if rectified_tf:
            mean = torch.load('laboratory/mean.pt').to(output.device)
            std = torch.load('laboratory/std.pt').to(output.device)
            noise = torch.randn_like(output).to(output.device)
            output -= mean
        # Step 4: Decode the output
        video = self._decode_output_to_video(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(
                    f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _ref_cache_token_len(self, conditional_dict):
        if conditional_dict is None:
            return 0
        ref_latents = conditional_dict.get("ref_latents", None)
        if ref_latents is None:
            return 0
        in_ch_need = self.generator.model.patch_embedding.in_channels
        if ref_latents.ndim == 4:
            ref_latents = ref_latents.unsqueeze(0).unsqueeze(3)
        elif ref_latents.ndim == 5 and ref_latents.shape[2] == in_ch_need:
            ref_latents = ref_latents.unsqueeze(3)
        elif ref_latents.ndim == 5:
            ref_latents = ref_latents.unsqueeze(0)
        if ref_latents.ndim != 6:
            return 0
        _, num_slots, _, ref_t, ref_h, ref_w = ref_latents.shape
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        return (
            int(num_slots)
            * (int(ref_t) // int(patch_t))
            * (int(ref_h) // int(patch_h))
            * (int(ref_w) // int(patch_w))
        )

    def _kv_cache_needs_reinit(self, ref_cache_token_len):
        if self.kv_cache1 is None:
            return True
        if getattr(self, "kv_cache_extra_tokens", 0) != int(ref_cache_token_len):
            return True
        return any("ref_token_len" not in cache for cache in self.kv_cache1)

    def _initialize_kv_cache(self, batch_size, dtype, device, extra_tokens=0):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            print("self.local_attn_size:", self.local_attn_size)
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = self.args.image_or_video_shape[1] * self.frame_seq_length
        kv_cache_size = kv_cache_size + int(extra_tokens)

        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "ref_token_len": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache
        self.kv_cache_extra_tokens = int(extra_tokens)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
