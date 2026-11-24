"""Run the Iris DMD teacher with its original dense score attention."""
import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KEYS = 'WASDIJKL'
REFERENCE_SLOTS = ('head', 'front', 'left', 'right', 'back')
TEACHER_NEGATIVE_PROMPT = (
    '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，'
    '最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，'
    '画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，'
    '杂乱的背景，三条腿，背景人很多，倒着走'
)


def negative_conditions(positive, negative_text, zero_action):
    return {
        'first_frame_latents': positive['first_frame_latents'],
        'prompt_embeds': negative_text,
        'act_context': zero_action,
        'act_context_scale': 0.0,
    }


def guided_flow(teacher, latent, timestep, positive, negative, scale):
    positive_flow = teacher(latent, positive, timestep)[0]
    if negative is None:
        return positive_flow
    negative_flow = teacher(latent, negative, timestep)[0]
    return (negative_flow.float() + scale *
            (positive_flow.float() - negative_flow.float())).to(positive_flow.dtype)


def read_conditions(action_path, prompt_path, frames, fps):
    action = json.loads(action_path.read_text())
    if action['fps'] != fps or action['total_frames'] != frames or len(action['frames']) != frames:
        raise ValueError('action.json must have one sample per output frame at the requested fps')
    rows = []
    for frame in action['frames']:
        keys = frame['keys']
        if set(keys) != set(KEYS) or any(type(value) is not bool for value in keys.values()):
            raise ValueError('Every action row requires eight boolean WASDIJKL keys')
        rows.append([int(keys[key]) for key in KEYS])
    prompt = json.loads(prompt_path.read_text())
    if (prompt.get('mode') != 'chunkwise' or prompt.get('num_frames') != frames
            or not prompt.get('frame_ranges')):
        raise ValueError('Expected a chunkwise prompt file covering all output frames')
    ranges = prompt['frame_ranges']
    if any(len(prompt.get(key, [])) != len(ranges) or
           any(not isinstance(s, str) for s in prompt[key])
           for key in ('global_prompts', 'action_prompts')):
        raise ValueError('Each range needs one global prompt and one action prompt')
    end = 0
    for lo, hi in ranges:
        if type(lo) is not int or type(hi) is not int or lo != end or not lo < hi <= frames:
            raise ValueError('Prompt ranges must be contiguous, ordered, and half-open')
        end = hi
    if end != frames:
        raise ValueError('Prompt ranges do not cover the last frame')
    return rows, prompt


def read_example(example, skill, frames):
    from abot_world.common.examples import EXAMPLE_SCENES, SKILL_BLOCKS

    scene = tuple(EXAMPLE_SCENES.values())[example - 1]
    # Skip the anchor and the first twelve-frame generation block.
    skill_end = 13 + 12 * SKILL_BLOCKS
    ranges = [[0, 13], [13, skill_end], [skill_end, frames]] if skill else [[0, frames]]
    prompt = {'mode': 'chunkwise', 'num_frames': frames, 'frame_ranges': ranges,
              'global_prompts': [scene.scene_prompt] * len(ranges),
              'action_prompts': ['', scene.skills[skill - 1].prompt, ''] if skill else ['']}
    return scene.first_frame, scene.reference_dir, [[1, 0, 0, 0, 0, 0, 0, 0] for _ in range(frames)], prompt


def main():
    from abot_world.common.examples import EXAMPLE_SCENES, SKILLS_PER_EXAMPLE

    p = argparse.ArgumentParser(
        description='Run the bidirectional Iris-BiT teacher. Only absolute RoPE is supported.',
        allow_abbrev=False,
    )
    p.add_argument('--teacher-dir', type=Path, default=ROOT / 'checkpoints/ABot-World-0-5-5B-Iris-BiT')
    p.add_argument('--shared-assets', type=Path, default=ROOT / 'checkpoints/ABot-World-0-5-5B-Iris')
    p.add_argument('--example', type=int, choices=range(1, len(EXAMPLE_SCENES) + 1),
                   help='Use a built-in third-person scene')
    p.add_argument('--skill', type=int, choices=range(1, SKILLS_PER_EXAMPLE + 1),
                   help='Cast one built-in skill for four chunks')
    for flag in ('first-frame', 'references', 'action', 'prompt', 'output'):
        p.add_argument('--' + flag, type=Path)
    p.add_argument('--check-weights', action='store_true')
    p.add_argument('--frames', type=int, choices=(121, 321), default=121)
    p.add_argument('--steps', type=int, default=48, help='Denoising steps (default: 48)')
    p.add_argument('--guidance-scale', type=float, default=3.0,
                   help='CFG scale (default: 3.0); 1 disables the negative pass')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--accept-teacher-absolute-rope', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.check_weights:
        if args.steps < 1:
            p.error('Inference requires positive --steps')
        if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
            p.error('--guidance-scale must be finite and at least 1')
        if args.output is None:
            p.error('Inference requires --output')
        if args.example:
            if any(getattr(args, x) is not None for x in ('first_frame', 'references', 'action', 'prompt')):
                p.error('--example uses its own image, references, actions and prompt')
            args.first_frame, args.references, action, prompt = read_example(args.example, args.skill, args.frames)
        else:
            if args.skill or any(getattr(args, x) is None for x in ('first_frame', 'action', 'prompt')):
                p.error('Use --example with optional --skill, or supply --first-frame, --action and --prompt')
            action, prompt = read_conditions(args.action, args.prompt, args.frames, 16)
        if args.output.exists():
            p.error('Output directory already exists')
    import torch
    from abot_world.bit.utils.teacher import BidirectionalTeacher

    teacher = BidirectionalTeacher(
        config_dir=str(args.teacher_dir.resolve()),
        checkpoint=str(args.teacher_dir / 'teacher.safetensors'),
    ).eval().requires_grad_(False)
    if args.check_weights:
        print(json.dumps({'strict_load': True, 'tensors': len(teacher.model.state_dict()),
                          'teacher_attention': 'bidirectional', 'relative_rope': False}))
        return

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageOps

    from abot_world.bit.utils.actions import pack_action_context
    from abot_world.bit.utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

    device, dtype = torch.device('cuda'), torch.bfloat16
    height, width = 704, 1280
    assets = args.shared_assets

    def image_tensor(path, w, h):
        with Image.open(path) as img:
            img = ImageOps.fit(img.convert('RGB'), (w, h), method=Image.Resampling.LANCZOS)
            return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).to(
                device=device, dtype=dtype)[None, :, None].div(127.5).sub(1)

    with torch.inference_mode():
        encoder = WanTextEncoder(tokenizer_path=str(assets / 'google/umt5-xxl'),
                                 encoder_pth_path=str(assets / 'models_t5_umt5-xxl-enc-bf16.pth'),
                                 dtype=dtype).to(device).eval()
        conditions = encoder(prompt, chunk_global_token_budget=384, chunk_action_token_budget=128)
        negative_text = (encoder([TEACHER_NEGATIVE_PROMPT])['prompt_embeds']
                         if args.guidance_scale > 1 else None)
        del encoder
        torch.cuda.empty_cache()
        vae = WanVAEWrapper(pretrained_path=str(assets / 'Wan2.2_VAE.pth'),
                            vae_type='Wan2.2_VAE').to(device=device, dtype=dtype).eval()
        anchor = vae.encode_to_latent(image_tensor(args.first_frame, width, height)).to(dtype)
        refs = anchor.new_zeros((1, 5, 48, 1, 32, 32))
        mask = anchor.new_zeros((1, 5))
        if args.references:
            for i, slot in enumerate(REFERENCE_SLOTS):
                candidates = [args.references / (slot + ext) for ext in ('.png', '.jpg', '.jpeg')]
                found = [path for path in candidates if path.is_file()]
                if len(found) > 1:
                    raise ValueError(f'Ambiguous reference image for {slot}')
                if found:
                    refs[:, i] = vae.encode_to_latent(image_tensor(found[0], 512, 512)).transpose(1, 2)
                    mask[:, i] = 1
            if not mask.any().item():
                raise ValueError('The reference directory has no named reference images')
        conditions.update(first_frame_latents=anchor, ref_latents=refs, ref_mask=mask,
                          act_context=pack_action_context(action, height=height, width=width,
                                                          device=device, dtype=dtype))
        negative = None
        if negative_text is not None:
            zero_action = pack_action_context([[0] * len(KEYS)] * args.frames,
                                              height=height, width=width, device=device, dtype=dtype)
            negative = negative_conditions(conditions, negative_text, zero_action)
            del negative_text, zero_action
        vae.to('cpu')
        torch.cuda.empty_cache()
        teacher.to(device=device, dtype=dtype)
        scheduler = teacher.get_scheduler()
        scheduler.set_timesteps(args.steps)
        length = (args.frames - 1) // 4 + 1
        rng = torch.Generator(device=device).manual_seed(args.seed)
        latent = torch.randn((1, length, *anchor.shape[2:]), device=device, dtype=dtype, generator=rng)
        for index, step in enumerate(scheduler.timesteps.tolist()):
            ts = torch.full((1, length), step, device=device, dtype=torch.float32)
            flow = guided_flow(teacher, latent, ts, conditions, negative, args.guidance_scale)
            latent = scheduler.step(flow.flatten(0, 1), ts.flatten(),
                                    latent.flatten(0, 1)).unflatten(0, (1, length)).to(dtype)
            latent[:, :1] = anchor
            if not torch.isfinite(latent).all().item():
                raise RuntimeError(f'Nonfinite latent at denoising iteration {index}')
        del teacher, flow, conditions, negative
        torch.cuda.empty_cache()
        vae.to(device)
        pixels = vae.decode_to_pixel(latent)
        if tuple(pixels.shape) != (1, args.frames, 3, height, width) or not torch.isfinite(pixels).all().item():
            raise RuntimeError('Decoded output has an invalid shape or nonfinite values')
        args.output.mkdir(parents=True, exist_ok=False)
        video = ((pixels[0].permute(0, 2, 3, 1) + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        with imageio.get_writer(args.output / 'video.mp4', fps=16, codec='libx264') as writer:
            for frame in video:
                writer.append_data(frame)
        (args.output / 'result.json').write_text(json.dumps(
            {'frames': args.frames, 'fps': 16, 'seed': args.seed, 'denoising_steps': args.steps,
             'relative_rope': False,
             'sampler': 'flow-match-cfg' if args.guidance_scale > 1 else 'flow-match-conditional-only',
             'cfg_scale': args.guidance_scale,
             'reference_slots': list(REFERENCE_SLOTS) if args.references else [],
             'negative_prompt': TEACHER_NEGATIVE_PROMPT if args.guidance_scale > 1 else None,
             'negative_action_scale': 0.0 if args.guidance_scale > 1 else None,
             'negative_references': False if args.guidance_scale > 1 else None},
            ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
