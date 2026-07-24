#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 项目根
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)
os.environ.setdefault("PROJECT_ROOT", str(_ROOT))

import imageio
import torch

from web_client.config import (
    KEY_ORDER,
    LATENT_CHANNELS,
    LATENT_HEIGHT,
    OUTPUT_DIR,
    LATENT_WIDTH,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    VIDEO_FPS,
)
from web_client.pipeline_loader import _init_lock, _get_pipeline_cached, decode_block_to_frames, get_pipeline

BENCH_REF_IMAGE = "web_client/datasets/images/example.png"
# Pre-generated ref images cache dir keyed by input image hash.
_ref_img_hash = hashlib.md5(Path(BENCH_REF_IMAGE).read_bytes()).hexdigest()[:16]
BENCH_REF_IMAGE_CACHE_DIR = str(_ROOT / "outputs" / "ref_image_cache" / _ref_img_hash)
QUANT_TYPES_TO_BENCH = [
    "none",
    "int8-torchao",
    "int8-triton",
    "int8-vllm",
    "int8-sgl",
    "int8-q8f",
    "mxfp6",
    "fp8-per-tensor",
    "fp8-per-tensor-weight-only",
    "fp8-per-block",
    "fp8-per-channel-vllm",
    "fp8-per-token",
    "fp8-per-token-sgl",
    "nvfp4",
    "mxfp4",
]

# Default per-frame action JSON (60 blocks ≈ 60 s, trimmed from the long-video dataset).
DEFAULT_ACTION_JSON = "web_client/datasets/actions/default_action.json"


def _parse_action_json(
    action_json_path: str,
    frames_per_block: int,
    video_fps: int,
) -> list[list[int]]:
    """Parse a per-frame action.json into per-block actions.

    The JSON format is::

        {"total_frames": 960, "fps": 16,
         "frames": [{"keys": {"W": true, ...}, "frame_id": "000000"}, ...]}

    One action per block is sampled by aligning block time to action.json time:
        block_i  →  frame int(i * action_fps * frames_per_block / video_fps)

    Returns one ``[W,A,S,D,I,J,K,L]`` entry per covered block.
    """
    with open(action_json_path, encoding="utf-8") as f:
        data = json.load(f)

    action_fps = int(data.get("fps", 16))
    frames_data = data["frames"]
    total_frames = int(data.get("total_frames", len(frames_data)))

    action_frames_per_block = action_fps * frames_per_block / video_fps
    num_blocks = max(int(total_frames / action_frames_per_block), 1)

    block_actions: list[list[int]] = []
    for block_idx in range(num_blocks):
        frame_idx = min(int(block_idx * action_frames_per_block), total_frames - 1)
        keys = frames_data[frame_idx]["keys"]
        block_actions.append([int(bool(keys.get(k, False))) for k in KEY_ORDER])

    print(
        f"[action] Parsed {len(block_actions)} block actions from {action_json_path} "
        f"(action_fps={action_fps}, total_frames={total_frames}, "
        f"frames_per_block={frames_per_block}, video_fps={video_fps})"
    )
    return block_actions


def _expand_action_list(
    action_list: list, total_blocks: int, mode: str = "repeat"
) -> list:
    """Expand *action_list* to cover *total_blocks*.

    ``repeat``: cycle through the list (suitable for long-video inference).
    ``fixed``:  truncate if longer, pad with last action if shorter.
    """
    if not action_list:
        return None
    n = len(action_list)
    if mode == "repeat":
        return [action_list[i % n] for i in range(total_blocks)]
    if total_blocks <= n:
        return action_list[:total_blocks]
    return action_list + [action_list[-1]] * (total_blocks - n)


def _build_failed_result(
    quant_type: str,
    phase: str,
    exc: Exception,
    error_stage: str = "run_single_quant",
) -> dict:
    return {
        "quant_type": quant_type,
        "status": "failed",
        "phase": phase,
        "error_stage": error_stage,
        "error_type": type(exc).__name__,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "runtime": {},
        "video_path": "",
    }


def _bytes_to_gb(v: int | float) -> float:
    return float(v) / (1024 ** 3)


def _write_video(frames: list, video_path: Path) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(video_path),
        fps=VIDEO_FPS,
        format="FFMPEG",
        codec="libx264",
        ffmpeg_params=[
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
        ],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _run_single_quant(args, quant_type: str, quant_dir: Path) -> dict:
    phase = str(args.phase).strip().lower()
    use_fp8_gemm = quant_type != "none"
    pipeline, _, device = get_pipeline(
        vae_type=args.vae_type,
        use_fp8_gemm=use_fp8_gemm,
        quant_type=quant_type,
    )
    dtype = torch.bfloat16
    num_fpb = int(pipeline.num_frame_per_block)
    blocks = max(int(args.fps_blocks), 1)

    _vae_for_shape = pipeline.encoder if pipeline.encoder is not None else pipeline.vae
    _vae_upsampling = getattr(_vae_for_shape, "upsampling_factor", 8)
    _latent_c = _vae_for_shape.z_dim
    _latent_h = STREAM_HEIGHT // _vae_upsampling
    _latent_w = STREAM_WIDTH // _vae_upsampling
    latent_shape = (1, num_fpb, _latent_c, _latent_h, _latent_w)
    frames_per_latent = 4
    frames_per_block = num_fpb * frames_per_latent
    action_template = {k: False for k in KEY_ORDER}
    action_template["W"] = True

    _prompt = "| unknown |"
    _ref_image_path = BENCH_REF_IMAGE
    _ref_cache_dir = BENCH_REF_IMAGE_CACHE_DIR
    # Build action list from --action-list (inline JSON) or --action-json (per-frame
    # action.json, default web_client/datasets/actions/default_action.json).
    _raw_action_list = getattr(args, "action_list", None)
    if _raw_action_list is not None:
        _parsed_actions = json.loads(_raw_action_list)
        _action_source = "--action-list"
    else:
        _action_source = getattr(args, "action_json", DEFAULT_ACTION_JSON)
        _parsed_actions = _parse_action_json(_action_source, frames_per_block, VIDEO_FPS)
    _action_mode = getattr(args, "action_mode", "repeat")
    _block_actions = _expand_action_list(_parsed_actions, blocks, _action_mode)
    print(
        f"[action] source={_action_source}, action_list_len={len(_parsed_actions)}, "
        f"mode={_action_mode}, expanded_to={len(_block_actions)} blocks "
        f"(target_blocks={blocks})"
    )

    def _reset_for_stream() -> None:
        print(">>> set_prompts start", flush=True)
        pipeline.set_prompts([_prompt], device=device)
        print(">>> set_prompts done", flush=True)
        print(">>> set_ref start", flush=True)
        pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=_ref_cache_dir, device=device)
        print(">>> set_ref done", flush=True)
        pipeline.reset_stream(1, dtype=dtype, device=device, initial_latent=None)

    def _run_stream_inference(input_blocks: int, decode: bool) -> dict:
        _reset_for_stream()
        lat_blocks: list[torch.Tensor] = []
        frames: list = []
        denoise_pred = None

        mem_alloc_sum = 0.0
        mem_reserved_sum = 0.0
        mem_sample_count = 0
        mem_peak_allocated = 0.0
        mem_peak_reserved = 0.0
        mem_start_allocated = float(torch.cuda.memory_allocated(device))
        mem_start_reserved = float(torch.cuda.memory_reserved(device))
        mem_end_allocated = mem_start_allocated
        mem_end_reserved = mem_start_reserved

        def _sample_memory() -> None:
            nonlocal mem_alloc_sum
            nonlocal mem_reserved_sum
            nonlocal mem_sample_count
            nonlocal mem_peak_allocated
            nonlocal mem_peak_reserved
            nonlocal mem_end_allocated
            nonlocal mem_end_reserved
            allocated = float(torch.cuda.memory_allocated(device))
            reserved = float(torch.cuda.memory_reserved(device))
            mem_alloc_sum += allocated
            mem_reserved_sum += reserved
            mem_sample_count += 1
            mem_peak_allocated = max(mem_peak_allocated, allocated)
            mem_peak_reserved = max(mem_peak_reserved, reserved)
            mem_end_allocated = allocated
            mem_end_reserved = reserved

        _sample_memory()

        for block_idx in range(input_blocks):
            noise_block = torch.randn(latent_shape, device=device, dtype=dtype)
            if block_idx == 0:
                pipeline.set_first_frame_latent(
                    _ref_image_path,
                    height=STREAM_HEIGHT,
                    width=STREAM_WIDTH,
                    device=device,
                )
            # Use per-block action, or fall back to action_template.
            # Supports both dict format {'W': True, ...} and list format [1, 0, 0, 0, 0, 0, 0, 0].
            if _block_actions is not None and block_idx < len(_block_actions):
                raw_action = _block_actions[block_idx]
                if isinstance(raw_action, dict):
                    block_action = raw_action
                else:
                    block_action = {k: bool(v) for k, v in zip(KEY_ORDER, raw_action)}
            else:
                block_action = action_template
            pipeline.set_act(
                block_action,
                height=STREAM_HEIGHT,
                width=STREAM_WIDTH,
                num_frames=num_fpb,
                device=device,
            )
            denoise_pred = pipeline.generate_next_block(noise_block)
            if denoise_pred is None:
                continue
            lat_blocks.append(denoise_pred)
            if decode:
                decoded_frames = decode_block_to_frames(pipeline, denoise_pred)
                frames.extend(decoded_frames)
            _sample_memory()

        _sample_memory()

        avg_allocated = mem_alloc_sum / max(mem_sample_count, 1)
        avg_reserved = mem_reserved_sum / max(mem_sample_count, 1)

        return {
            "lat_blocks": lat_blocks,
            "frames": frames,
            "block_count": len(lat_blocks),
            "frame_count": len(lat_blocks) * frames_per_block,
            "runtime": {
                "memory_gb": {
                    "start_allocated": _bytes_to_gb(mem_start_allocated),
                    "start_reserved": _bytes_to_gb(mem_start_reserved),
                    "end_allocated": _bytes_to_gb(mem_end_allocated),
                    "end_reserved": _bytes_to_gb(mem_end_reserved),
                    "avg_allocated": _bytes_to_gb(avg_allocated),
                    "avg_reserved": _bytes_to_gb(avg_reserved),
                    "peak_allocated": _bytes_to_gb(mem_peak_allocated),
                    "peak_reserved": _bytes_to_gb(mem_peak_reserved),
                    "sample_count": mem_sample_count,
                },
            },
        }

    e2e_run = _run_stream_inference(input_blocks=blocks, decode=True)
    print(f"[Run] blocks={e2e_run['block_count']}, frames={e2e_run['frame_count']}")
    runtime_result = e2e_run.get("runtime", {})
    mem_runtime = runtime_result.get("memory_gb", {})
    print(
        "[Memory] allocated(avg/peak)={avg_alloc:.3f}/{peak_alloc:.3f} GB, "
        "reserved(avg/peak)={avg_res:.3f}/{peak_res:.3f} GB".format(
            avg_alloc=float(mem_runtime.get("avg_allocated", 0.0)),
            peak_alloc=float(mem_runtime.get("peak_allocated", 0.0)),
            avg_res=float(mem_runtime.get("avg_reserved", 0.0)),
            peak_res=float(mem_runtime.get("peak_reserved", 0.0)),
        )
    )

    video_path = quant_dir / f"{quant_type}.mp4"
    if e2e_run["frames"]:
        _write_video(e2e_run["frames"], video_path)
    else:
        video_path = Path("")

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    result = {
        "quant_type": quant_type,
        "status": "ok",
        "phase": phase,
        "runtime": runtime_result,
        "video_path": str(video_path) if str(video_path) else "",
        "meta": {
            "num_frame_per_block": num_fpb,
            "frames_per_latent": frames_per_latent,
            "frames_per_block": frames_per_block,
            "fps_blocks_input": blocks,
        },
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline streaming inference (DiT + VAE decode)")
    ap.add_argument("--phase", choices=("dit", "vae", "both"), default="both", help="分析阶段")
    ap.add_argument(
        "--fps-blocks",
        type=int,
        default=3600,
        help="推理生成的 block 数 (默认 3600 ≈ 1 小时 @12fps×12frames/block)",
    )
    ap.add_argument("--prompt", type=str, default="| unknown |", help="文本条件")
    ap.add_argument(
        "--quant-type",
        type=str,
        default="all",
        help="传 all 自动遍历预设量化类型；传具体 quant_type 则仅测试一个",
    )
    ap.add_argument(
        "--action-json",
        type=str,
        default=DEFAULT_ACTION_JSON,
        help="Path to a per-frame action.json used when --action-list is not given. "
             f"Default: {DEFAULT_ACTION_JSON} (60-block ≈ 1-minute sequence).",
    )
    ap.add_argument(
        "--action-list",
        type=str,
        default=None,
        help="JSON-encoded list of per-block actions, each [W,A,S,D,I,J,K,L]; "
             "overrides --action-json. "
             "Example: '[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]]'",
    )
    ap.add_argument(
        "--action-mode",
        type=str,
        choices=("repeat", "fixed"),
        default="repeat",
        help="How to apply action_list when blocks > len(action_list): "
             "'repeat' cycles through the list (default, for long-video inference), "
             "'fixed' truncates or pads with the last action.",
    )
    ap.add_argument(
        "--vae_type",
        type=str,
        default=None,
        choices=["wan2.2", "taew2_2", "mg_lightvae", "mg_lightvae_v2"],
        help="VAE 类型（默认读 config）；taew2_2 需权重",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA GPU 与正确驱动。", file=sys.stderr)
        sys.exit(1)

    quant_arg = str(args.quant_type).strip().lower()
    quant_list = QUANT_TYPES_TO_BENCH if quant_arg == "all" else [quant_arg]

    run_dir = OUTPUT_DIR / "profile_bench" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for idx, quant_type in enumerate(quant_list, start=1):
        quant_dir = run_dir / quant_type
        quant_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(quant_list)}] quant_type={quant_type}")
        print("=" * 100)
        try:
            item = _run_single_quant(args, quant_type, quant_dir)
        except KeyboardInterrupt:
            print("[abort] user interrupted.")
            raise
        except Exception as exc:
            item = _build_failed_result(
                quant_type=quant_type,
                phase=args.phase,
                exc=exc,
                error_stage="run_single_quant",
            )
            print(f"[error] quant_type={quant_type} failed: {item['error']}")
            # 清理 CUDA 缓存，尽量保证后续 quant_type 还能继续执行
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception as cleanup_exc:
                    print(f"[warn] cuda cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}")
        finally:
            # `all` 模式会依次构建多个 pipeline；清理 lru_cache 避免模型常驻导致 OOM。
            if quant_arg == "all":
                try:
                    _get_pipeline_cached.cache_clear()
                except Exception as cache_exc:
                    print(f"[warn] pipeline cache_clear failed: {type(cache_exc).__name__}: {cache_exc}")
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception as cleanup_exc:
                    print(f"[warn] post-run cuda cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}")
        results.append(item)

        # 每个 quant_type 跑完后立刻落盘，避免中途异常导致日志丢失
        results_json_path = run_dir / "results.json"
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    results_json_path = run_dir / "results.json"

    print("\n" + "#" * 100)
    print("Inference finished.")
    print(f"- results.json: {results_json_path}")
    print("#" * 100)


if __name__ == "__main__":
    main()
