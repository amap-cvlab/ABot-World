"""Iris teacher-only subset of the TF-route model wrappers."""
from typing import List
import types
import torch

from abot_world.bit.utils.scheduler import SchedulerInterface
from abot_world.bit.wan.modules.t5 import umt5_xxl
from abot_world.bit.wan.modules.tokenizers import HuggingfaceTokenizer


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        tokenizer_path="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/",
        encoder_pth_path="wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        # tokenizer_path = _resolve_wan_path_with_dir(tokenizer_path, wan_models_dir)
        # encoder_pth_path = _resolve_wan_path_with_dir(encoder_pth_path, wan_models_dir)

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(encoder_pth_path, map_location='cpu', weights_only=True)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def _encode(self, text_prompts: List[str], max_length: int = 512) -> torch.Tensor:
        ids, mask = self.tokenizer(
            text_prompts,
            return_mask=True,
            add_special_tokens=True,
            max_length=int(max_length),
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        context = self.text_encoder(ids, mask)
        context = context * mask.unsqueeze(-1).to(dtype=context.dtype)
        return context

    def _encode_chunkwise(
        self,
        prompt: dict,
        *,
        global_token_budget: int,
        action_token_budget: int,
    ) -> dict:
        frame_ranges = prompt.get("frame_ranges")
        num_frames = prompt.get("num_frames")
        global_prompts = prompt.get("global_prompts")
        action_prompts = prompt.get("action_prompts")

        if int(global_token_budget) + int(action_token_budget) != 512:
            raise ValueError(
                "chunk prompt token budgets must sum to 512, got "
                f"{global_token_budget}+{action_token_budget}"
            )

        if isinstance(global_prompts, list) and isinstance(action_prompts, list):
            if not global_prompts or len(global_prompts) != len(action_prompts):
                raise ValueError(
                    "chunk-wise prompt requires matching non-empty "
                    "global_prompts/action_prompts"
                )
            unique_globals = []
            global_lookup = {}
            global_inverse = []
            for value in global_prompts:
                value = str(value or "").strip()
                if value not in global_lookup:
                    global_lookup[value] = len(unique_globals)
                    unique_globals.append(value)
                global_inverse.append(global_lookup[value])
            global_context = self._encode(
                unique_globals, max_length=int(global_token_budget)
            )
            inverse = torch.as_tensor(
                global_inverse, dtype=torch.long, device=global_context.device
            )
            global_context = global_context.index_select(0, inverse)
            action_context = self._encode(
                [str(value or "").strip() for value in action_prompts],
                max_length=int(action_token_budget),
            )
            context = torch.cat([global_context, action_context], dim=1)
            prompt_count = len(global_prompts)
        else:
            combined_prompts = prompt.get("prompts")
            if not isinstance(combined_prompts, list) or not combined_prompts:
                raise ValueError("chunk-wise prompt requires a non-empty prompts list")
            context = self._encode(
                [str(value or "") for value in combined_prompts], max_length=512
            )
            prompt_count = len(combined_prompts)

        ranges = torch.as_tensor(frame_ranges, dtype=torch.long, device=self.device)
        if ranges.ndim != 2 or ranges.shape != (prompt_count, 2):
            raise ValueError(
                "chunk-wise prompt requires one [start,end] range per prompt; "
                f"got {tuple(ranges.shape)} for {prompt_count} prompts"
            )
        if num_frames is None or int(num_frames) <= 0:
            raise ValueError("chunk-wise prompt requires a positive num_frames")
        return {
            "prompt_embeds": context.unsqueeze(0),
            "cross_attn_frame_ranges": ranges.unsqueeze(0),
            "cross_attn_num_frames": int(num_frames),
        }

    def forward(
        self,
        text_prompts,
        *,
        chunk_global_token_budget: int = 384,
        chunk_action_token_budget: int = 128,
    ) -> dict:
        if isinstance(text_prompts, dict):
            if text_prompts.get("mode") != "chunkwise":
                raise ValueError(
                    f"unsupported prompt dictionary mode: {text_prompts.get('mode')!r}"
                )
            return self._encode_chunkwise(
                text_prompts,
                global_token_budget=chunk_global_token_budget,
                action_token_budget=chunk_action_token_budget,
            )
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        return {
            "prompt_embeds": self._encode(list(text_prompts), max_length=512)
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        pretrained_path="wan_models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth",
        z_dim=16,
        vae_type="Wan2.1_VAE",
        wan_models_dir=None,
    ):
        super().__init__()
        if vae_type == "Wan2.1_VAE":
            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)

            # init model
            from abot_world.bit.wan.modules.vae2_1 import _video_vae

            self.model = _video_vae(
                pretrained_path=pretrained_path,
                z_dim=z_dim,
            ).eval().requires_grad_(False)
        
        #TODO: Wan2.2_VAE
        elif vae_type == "Wan2.2_VAE":
            from abot_world.bit.wan.modules.vae2_2 import _video_vae
            # from wan.modules.vae2_2 import Wan2_2_VA
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

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
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
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
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
