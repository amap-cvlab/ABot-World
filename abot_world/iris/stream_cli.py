"""Standalone streaming inference: explicit model inputs to an MP4."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

from abot_world.iris.config import DEFAULT_STREAM_CONFIG
from abot_world.iris.utils.action_conditions import ActionConditions, CONTROL_TEXT

def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_STREAM_CONFIG)
    parser.add_argument('--first-frame', required=True)
    parser.add_argument('--references', required=True, help='Directory containing head/front/left/right/back images')
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument('--global-prompt')
    text.add_argument('--prompt-embeddings', help='Tensor saved by torch.save: [1,512,D]')
    parser.add_argument('--local-prompt', default='')
    parser.add_argument('--keys', nargs='*', choices=list('WASDIJKL'), default=[])
    parser.add_argument('--action', action='append', default=[], metavar='BLOCK:KEY[+KEY]',
                        help='At a zero-based block, change held action keys; use idle to release')
    parser.add_argument('--blocks', type=int, default=40)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--crf', type=int, default=28)
    parser.add_argument('--trace', action='store_true', help='Write per-block tensor hashes and synchronized timings')
    parser.add_argument('--output', required=True, help='New output directory; existing paths are rejected')
    args = parser.parse_args()
    if args.blocks < 1 or args.fps < 1:
        parser.error('blocks and fps must be positive')
    if args.action and args.prompt_embeddings:
        parser.error('--action requires --global-prompt')
    schedule = {}
    for spec in args.action:
        block, sep, key_text = spec.partition(':')
        keys_for_block = set() if key_text == 'idle' else set(key_text.split('+'))
        if (not sep or not block.isascii() or not block.isdecimal()
                or int(block) >= args.blocks or int(block) in schedule
                or not keys_for_block <= CONTROL_TEXT.keys()):
            parser.error(f'invalid or duplicate action {spec!r}; use BLOCK:KEY[+KEY] or BLOCK:idle')
        schedule[int(block)] = keys_for_block
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import imageio
    from omegaconf import OmegaConf
    from abot_world.iris.utils.inference_loader import load_streaming_model
    from abot_world.iris.utils.misc import set_seed
    with torch.inference_mode():
        pipeline, config, device = load_streaming_model(args.config)
        OmegaConf.save(config, output/'effective-config.yaml')
        fpb = int(config.num_frame_per_block)
        _, _, channels, latent_h, latent_w = config.image_or_video_shape
        height, width = latent_h * 16, latent_w * 16
        # Match the deployment path's generation RNG boundary, after loading.
        set_seed(int(config.get('seed', 42)))
        noise = torch.randn([1, fpb, channels, latent_h, latent_w], device=device, dtype=torch.bfloat16)
        pipeline.reset_stream(1, torch.bfloat16, device)
        if args.prompt_embeddings:
            pipeline.set_prompt_embeddings(torch.load(args.prompt_embeddings, map_location='cpu', weights_only=True))
        else:
            pipeline.set_condition_text(args.global_prompt, args.local_prompt)
        pipeline.set_ref_latent_mask_from_exists_paths(args.references, device)
        pipeline.set_first_frame_latent(args.first_frame, height, width, device)
        keys = dict.fromkeys(args.keys, True)
        actions = ActionConditions()
        held = set()
        current_local = args.local_prompt
        frames = 0
        trace_file = (output/'trace.jsonl').open('x') if args.trace else nullcontext()
        condition_names = ('prompt_embeds', 'ref_latents', 'ref_mask', 'first_frame_latents')
        with trace_file as trace, imageio.get_writer(
            str(output/'origin.mp4'), fps=args.fps, codec='libx264', quality=None,
            ffmpeg_params=['-crf', str(args.crf), '-preset', 'medium'],
        ) as writer:
            if trace is not None:
                static = {name: digest(pipeline.conditional_dict[name]) for name in condition_names}
            for block in range(args.blocks):
                next_held = schedule.get(block, held)
                body, caption = actions.step(next_held, next_held - held)
                held = next_held
                local = caption or args.local_prompt
                if local != current_local:
                    pipeline.set_condition_text(args.global_prompt, local)
                    if trace is not None:
                        static['prompt_embeds'] = digest(pipeline.conditional_dict['prompt_embeds'])
                    current_local = local
                if block:
                    pipeline.set_first_frame_latent(None, height, width, device)
                    if trace is not None:
                        static['first_frame_latents'] = None
                pipeline.set_act(keys, height=height, width=width, num_frames=fpb, device=device)
                if trace is not None:
                    row = dict(block=block, action=body, noise=digest(noise), rng=digest(torch.cuda.get_rng_state()),
                               conditions={**static, 'act_context': digest(pipeline.conditional_dict['act_context'])})
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                latent = pipeline.generate_next_block(noise)
                if trace is not None:
                    torch.cuda.synchronize()
                    row.update(output=digest(latent), shape=list(latent.shape), generation_seconds=time.perf_counter()-started)
                noise = torch.randn_like(noise)
                frames += pipeline.decode_block_and_write(latent, writer)
                if trace is not None:
                    trace.write(json.dumps(row)+'\n')
                    trace.flush()
                print(f'block={block+1}/{args.blocks} frames={frames}', flush=True)
            frames += pipeline.decode_block_and_write(None, writer, final=True)
        (output/'result.json').write_text(json.dumps(dict(blocks=args.blocks, frames=frames, fps=args.fps), indent=2))

if __name__ == '__main__':
    main()
