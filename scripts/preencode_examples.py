"""Build the optional Gradio text cache with the release UMT5 encoder."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abot_world.iris.utils.action_conditions import ACTION_COMBINATIONS, CONTROL_TEXT
from abot_world.iris.utils.text_embedding_cache import CACHE_VERSION, DEFAULT_TEXT_CACHE, TOKEN_LENGTHS, text_encoder_identity
from abot_world.common.examples import EXAMPLE_SCENES
from abot_world.iris.config import CONFIG_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_TEXT_CACHE,
                        help=f"New cache file (default: {DEFAULT_TEXT_CACHE})")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    from omegaconf import OmegaConf
    import torch
    from abot_world.iris.utils.runtime_memory import cpu_encoder_batch
    from abot_world.iris.utils.wan_wrapper import WanTextEncoder

    options = OmegaConf.load(CONFIG_DIR / "long_forcing_dmd.yaml").text_encoder_kwargs
    encoder_paths = {key: str(ROOT / value) for key, value in options.items()}
    identity = text_encoder_identity(**encoder_paths)
    encoder = WanTextEncoder(**encoder_paths).to(args.device).eval()
    scenes = list(EXAMPLE_SCENES.values())
    texts = {
        "global": [scene.scene_prompt for scene in scenes],
        "local": ["", *CONTROL_TEXT.values(), *ACTION_COMBINATIONS.values(),
                  *(skill.prompt for scene in scenes for skill in scene.skills)],
    }
    cache = {"version": CACHE_VERSION, "identity": identity}
    for group, length in TOKEN_LENGTHS.items():
        rows = {}
        cpu_batch = cpu_encoder_batch(encoder) if args.device == "cpu" else nullcontext()
        with torch.no_grad(), cpu_batch:
            prompts = tuple(dict.fromkeys(texts[group]))
            for index, prompt in enumerate(prompts, 1):
                row = encoder([prompt], device=args.device, max_length=length)["prompt_embeds"]
                row = row.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                if row.shape != (1, length, 4096) or not torch.isfinite(row).all():
                    raise ValueError(f"Invalid {group} embedding at row {index}")
                rows[prompt] = row
                print(f"{group}: {index}/{len(prompts)}", flush=True)
        cache[group] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".pt") as staging:
        torch.save(cache, staging)
        staging.flush()
        os.link(staging.name, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
