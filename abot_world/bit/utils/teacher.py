"""Load Iris-BiT weights and run bidirectional, action-conditioned inference."""

import torch
from safetensors.torch import load_file

from abot_world.bit.utils.wan_wrapper import WanDiffusionWrapper
from abot_world.bit.utils.scheduler import FlowMatchScheduler
from abot_world.bit.wan.modules.backbone import (
    WanBackboneSelfAttention, WanVideoBackbone, freqs_at, rope_apply_with_refimg,
)
from abot_world.bit.wan.modules.model import rope_apply
from abot_world.bit.wan.modules.attention import flash_attention


class BidirectionalTFScoreAttention(WanBackboneSelfAttention):
    def forward(self, x, seq_lens, grid_sizes, freqs, **kwargs):
        if kwargs.get("kv_cache") is not None:
            raise ValueError("score attention cannot use a causal KV cache")
        b, s = x.shape[:2]
        q = self.norm_q(self.q(x)).view(b, s, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(b, s, self.num_heads, self.head_dim)
        v = self.v(x).view(b, s, self.num_heads, self.head_dim)
        r = getattr(self, "_num_ref_tokens", 0)
        vq = rope_apply(q[:, r:], grid_sizes, freqs).type_as(v)
        vk = rope_apply(k[:, r:], grid_sizes, freqs).type_as(v)
        if r:
            rh, rw = self._ref_grid_sizes
            slots = r // (rh * rw)
            fdim = freqs.shape[1]
            dims = (fdim - 2 * (fdim // 3), fdim // 3, fdim // 3)
            axes = freqs.split(dims, dim=1)
            temporal = freqs_at([-(slots-i) * max(rh*rw, 256) for i in range(slots)],
                                2*dims[0], device=x.device)
            ref_freqs = torch.cat([
                temporal[:, None, None, :].expand(slots, rh, rw, -1),
                axes[1][:rh][None, :, None, :].expand(slots, rh, rw, -1),
                axes[2][:rw][None, None, :, :].expand(slots, rh, rw, -1),
            ], dim=-1).reshape(r, 1, -1)
            rq = rope_apply_with_refimg(q[:, :r], ref_freqs, self.num_heads).type_as(v)
            rk = rope_apply_with_refimg(k[:, :r], ref_freqs, self.num_heads).type_as(v)
            ref_out = flash_attention(rq, rk, v[:, :r])
            video_out = flash_attention(vq, torch.cat([rk, vk], dim=1), v)
            out = torch.cat([ref_out, video_out], dim=1)
        else:
            out = flash_attention(vq, vk, v)
        return self.o(out.flatten(2))


class BidirectionalTFScore(WanVideoBackbone):
    """Same parameterization/ref/text routing as TF, with bidirectional video."""

    _uses_dense_score_attention = True

    def _forward_train(self, x, t, context, seq_len, **kwargs):
        if kwargs.get("clean_x") is not None:
            raise ValueError("a score model must not receive a TF clean branch")
        return super()._forward_train(x, t, context, seq_len, **kwargs)


def normalize_checkpoint(state):
    for container in ("generator", "model", "state_dict"):
        if container in state and isinstance(state[container], dict):
            state = state[container]
            break
    result = {}
    for key, value in state.items():
        for wrapper in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
            key = key.replace(wrapper, "")
        if key.startswith("model."):
            key = key[len("model."):]
        if key in result:
            raise ValueError(f"duplicate checkpoint key: {key}")
        result[key] = value
    return result


class BidirectionalTeacher(WanDiffusionWrapper):
    def __init__(self, config_dir, checkpoint, *, block_size=3):
        super().__init__()
        config = WanVideoBackbone.load_config(config_dir)
        if int(config.get("sink_size", 0)) != 0:
            raise ValueError("TF-route rollout currently requires sink_size=0; cache has no extra sink capacity")
        # Model geometry comes from the teacher config, never from 1.3B defaults.
        self.model = BidirectionalTFScore.from_config(
            config, model_type="ti2v", use_relative_rope=False,
            independent_first_frame=True, num_frame_per_block=block_size,
            local_attn_size=-1,
        )
        self.model._train_dmd_action_adapter = True
        if self.model.act_control_adapter is not None:
            self.model.act_control_adapter.requires_grad_(True)
        for block in self.model.blocks:
            block.self_attn = BidirectionalTFScoreAttention(
                self.model.dim, self.model.num_heads, qk_norm=self.model.qk_norm,
                eps=self.model.eps, use_relative_rope=False)
        if checkpoint is not None:
            state = load_file(checkpoint, device="cpu") if checkpoint.endswith(".safetensors") else torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(normalize_checkpoint(state), strict=True)
            del state
        self.model.eval()
        self.model_type = "ti2v"
        self.uniform_timestep = False
        self.seq_len = None
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()

    def forward(self, noisy_image_or_video, conditional_dict, timestep):
        anchor = conditional_dict["first_frame_latents"]
        noisy_image_or_video = torch.cat([anchor, noisy_image_or_video[:, 1:]], dim=1)
        timestep = timestep.clone()
        timestep[:, :1] = 0
        flow = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4), t=timestep,
            context=conditional_dict["prompt_embeds"], seq_len=None,
            act_context=conditional_dict.get("act_context"),
            act_context_scale=conditional_dict.get("act_context_scale", 1.0),
            ref_latents=conditional_dict.get("ref_latents"),
            ref_mask=conditional_dict.get("ref_mask"),
            cross_attn_frame_ranges=conditional_dict.get("cross_attn_frame_ranges"),
            cross_attn_num_frames=conditional_dict.get("cross_attn_num_frames"),
        ).permute(0, 2, 1, 3, 4)
        pred = self._convert_flow_pred_to_x0(
            flow.flatten(0, 1), noisy_image_or_video.flatten(0, 1), timestep.flatten(),
        ).unflatten(0, timestep.shape)
        return flow, torch.cat([anchor, pred[:, 1:]], dim=1)
