"""
web_client/inference.py — 推理 worker、流式生成控制、Gradio 回调。
"""
import hashlib
import time
import numpy as np
import queue
import threading
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
import gradio as gr
from PIL import Image

from web_client.config import (
    STREAM_HEIGHT, STREAM_WIDTH, DEFAULT_REF_IMAGE, DEBUG_FRONTEND_ONLY,
    LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH,
    FRAME_QUEUE_SIZE, QUEUE_POLL_TIMEOUT, WORKER_JOIN_TIMEOUT,
    VAE_TYPE, USE_FP8_GEMM, OUTPUT_DIR, VIDEO_FPS, VIDEO_CODEC, VIDEO_QUALITY,
    MAX_BLOCKS, FRAMES_PER_BLOCK, FIRST_BLOCK_FRAMES,
)
from web_client.state import state
from web_client.pipeline_loader import get_pipeline, decode_block_to_frames, _init_lock
from web_client.ui_helpers import format_current_prompt_md, format_status_md, format_progress_bar_html

_SENTINEL = None  # worker 结束信号
MIN_SLEEP = 0.005  # 5ms: 小于该值的 sleep 不执行，以减少调度抖动


# ── Worker ───────────────────────────────────────────────────────────────────

def _inference_worker(pipeline, noise, device, frame_queue, num_fpb):
    """Runs in a background thread: generates blocks and pushes
    (frames, block_elapsed) tuples into frame_queue.
    每个 block 通过 set_act(..., height, width, num_frames, device) 更新动作条件后生成。
    """
    from web_client.config import MAX_BLOCKS
    from web_client.keyboard import _sample_key_snapshot  # 避免循环导入
    current_prompt = state.shared_prompt

    try:
        while state.is_running and state.block_count < MAX_BLOCKS:
            sp = state.shared_prompt

            key_snapshot = _sample_key_snapshot()

            if sp != current_prompt:
                current_prompt = sp
                pipeline.set_prompts([current_prompt], device=device)
            block_start = time.monotonic()

            lat_block = None
            while lat_block is None:
                pipeline.set_act(
                    key_snapshot,
                    height=STREAM_HEIGHT, width=STREAM_WIDTH,
                    num_frames=num_fpb, device=device,
                )

                lat_block = pipeline.generate_next_block(noise)
                noise = torch.randn_like(noise)

            frames = decode_block_to_frames(pipeline, lat_block)
            block_end = time.monotonic()
            block_elapsed = block_end - block_start

            state.block_count += 1
            state.frame_count += len(frames)
            frame_queue.put((frames, block_elapsed, current_prompt, block_end))

    except Exception as e:
        state.worker_error = str(e)
    finally:
        # 判断完成状态：达到 MAX_BLOCKS 为正常完成，否则为被停止
        from web_client.config import MAX_BLOCKS
        if state.block_count >= MAX_BLOCKS:
            state.completion_status = "completed"
        else:
            state.completion_status = "stopped"
        frame_queue.put(_SENTINEL)


# ── Stream control ───────────────────────────────────────────────────────────


def _btn_update():
    """根据全局状态统一计算按钮的 value 与 interactive。"""
    if not state.model_ready and not DEBUG_FRONTEND_ONLY:
        return gr.update(value="唤醒你的世界", interactive=False)
    if state.is_stopping:
        return gr.update(value="停止中...", interactive=False)
    if state.is_running:
        return gr.update(value="重塑你的世界", interactive=not state.waiting_for_new_frames)
    return gr.update(value="唤醒你的世界", interactive=True)


def update_state(prompt: str, ref_image_path=None):
    """热刷新你的世界（流已在运行时）。"""
    state.shared_prompt = (prompt or "").strip()
    state.waiting_for_new_frames = True
    state.played_frame_count = 0
    status_data = state.get_status_data(state.shared_prompt)
    status_md = format_status_md(status_data)
    prompt_md = format_current_prompt_md(state.shared_prompt)
    return (
        status_md,
        prompt_md,
        _btn_update(),
    )


def start_stream(prompt: str, ref_image_path=None, vae_type: str | None = None, use_fp8_gemm: bool | None = None):
    """启动流式生成并逐帧 yield (frame, status, current_prompt_md, btn_update, overlay_visible)。"""
    prompt = prompt.strip()
    state.shared_prompt = prompt
    state.is_running = True
    state.played_frame_count = 0
    state.reset_stats()

    # 初始化变量（确保 except 块可访问）
    video_path = None
    thumbnail_path = None
    video_writer = None
    first_frame_saved = False
    first_frame_yielded = False  # 追踪是否已 yield 首帧，用于控制 overlay 隐藏
    total_frames = 0  # decode 后的实际帧数 * MAX_BLOCKS，在拿到首个 block 后计算

    # 创建视频输出目录和文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = str(OUTPUT_DIR / f"stream_{ts}.mp4")
    thumbnail_path = str(OUTPUT_DIR / f"stream_{ts}.png")
    # video_writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec=VIDEO_CODEC, quality=VIDEO_QUALITY)

    video_writer = imageio.get_writer(
        video_path,
        fps=VIDEO_FPS,
        format="FFMPEG",
        codec="libx264",
        ffmpeg_params=[
            "-crf", "18",          # 无损
            "-preset", "fast",# 更高压缩率，编码更慢
            "-pix_fmt", "yuv420p",  # 与 x264rgb 匹配
        ],
    )

    state.current_video_writer = video_writer
    state.current_video_path = video_path
    state.current_thumbnail_path = thumbnail_path

    _vae_type = VAE_TYPE if vae_type is None else str(vae_type)
    fp8 = USE_FP8_GEMM if use_fp8_gemm is None else bool(use_fp8_gemm)
    with _init_lock:
        pipeline, config, device = get_pipeline(
            vae_type=_vae_type,
            use_fp8_gemm=fp8,
        )

    num_fpb = int(getattr(
        config, "num_frame_per_block",
        getattr(config, "model_kwargs", {}).get("num_frame_per_block", 1),
    ))
    _vae_for_shape = pipeline.encoder if pipeline.encoder is not None else pipeline.vae
    vae_upsampling_factor = getattr(_vae_for_shape, "upsampling_factor", 8)
    latent_channels = _vae_for_shape.z_dim
    latent_height = STREAM_HEIGHT // vae_upsampling_factor
    latent_width = STREAM_WIDTH // vae_upsampling_factor
    noise = torch.randn([1, num_fpb, latent_channels, latent_height, latent_width], device=device, dtype=torch.bfloat16)

    state.ref_image_path = ref_image_path if ref_image_path else None
    final_ref_image = state.ref_image_path or DEFAULT_REF_IMAGE

    pipeline.torch_dtype = torch.bfloat16
    print(">>> set_prompts start", flush=True)
    pipeline.set_prompts([prompt], device=device)
    print(">>> set_prompts done", flush=True)
    ref_img_hash = hashlib.md5(Path(final_ref_image).read_bytes()).hexdigest()[:16]
    ref_cache_dir = OUTPUT_DIR / "ref_image_cache" / ref_img_hash
    print(f"[REF][CACHE] Using ref cache dir: {final_ref_image} -> {ref_cache_dir}", flush=True)
    print(">>> set_ref start", flush=True)
    pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=str(ref_cache_dir), device=device)
    print(">>> set_ref done", flush=True)
    pipeline.reset_stream(batch_size=1, dtype=torch.bfloat16, device=device, initial_latent=None)
    print(">>> set_first_frame_latent start", flush=True)
    pipeline.set_first_frame_latent(
        final_ref_image,
        height=STREAM_HEIGHT, width=STREAM_WIDTH, device=device,
    )
    print(">>> set_first_frame_latent done", flush=True)

    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    worker = threading.Thread(
        target=_inference_worker,
        args=(pipeline, noise, device, frame_queue, num_fpb),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            try:
                item = frame_queue.get(timeout=QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                if not state.is_running:
                    break
                continue

            if item is _SENTINEL:
                if getattr(state, "worker_error", None):
                    state.is_running = False
                    yield (
                        gr.skip(),
                        f"**错误**：{state.worker_error}",
                        format_current_prompt_md(state.shared_prompt),
                        _btn_update(),
                        gr.update(visible=False),
                        "",
                    )
                    state.worker_error = None
                break

            frames, block_elapsed, current_prompt, block_end = item
            # 首个 block 到达后，根据 decode 后的实际帧数计算总预期帧数
            if total_frames == 0:
                total_frames = FIRST_BLOCK_FRAMES + (MAX_BLOCKS - 1) * FRAMES_PER_BLOCK
            status_data = state.get_status_data(current_prompt)
            status_md = format_status_md(status_data)
            prompt_md = format_current_prompt_md(current_prompt)

            if state.waiting_for_new_frames:
                state.waiting_for_new_frames = False
            update_btn_state = _btn_update()

            frames_count = len(frames)
            if not frames_count:
                continue

            pk_start = time.monotonic()
            # 目标结束时刻 = 生成结束时间 + 生成耗时（预期下一块完成时刻）
            pk_end_target = block_end + block_elapsed

            for i, frame in enumerate(frames):
                # 写入视频文件（帧为 numpy 数组）
                if video_writer is not None:
                    try:
                        video_writer.append_data(np.array(frame))
                    except Exception:
                        pass

                # 保存首帧作为缩略图
                if not first_frame_saved:
                    try:
                        img = Image.fromarray(np.array(frame))
                        img.save(thumbnail_path)
                        first_frame_saved = True
                    except Exception:
                        pass

                # 每帧都换图，但 status / 按钮与上一帧相同时不必重复推送，减轻 Markdown DOM 整段替换导致的闪烁
                out_status = status_md if i == 0 else gr.skip()
                out_prompt = prompt_md if i == 0 else gr.skip()
                out_btn = update_btn_state if i == 0 else gr.skip()
                # 首帧到达时隐藏 overlay 标签
                if not first_frame_yielded:
                    first_frame_yielded = True
                    out_overlay = gr.update(visible=False)
                else:
                    out_overlay = gr.skip()
                # 更新播放进度
                state.played_frame_count += 1
                progress_html = format_progress_bar_html(state.played_frame_count, state.frame_count, total_frames)
                yield frame, out_status, out_prompt, out_btn, out_overlay, progress_html

                # 仅在帧与帧之间控制间隔：第一帧尽快推送
                if i < frames_count - 1:
                    alpha_next = (i + 1) / (frames_count - 1)
                    target_next = pk_start + alpha_next * (pk_end_target - pk_start)
                    sleep_time = target_next - time.monotonic()
                    if sleep_time > MIN_SLEEP:
                        time.sleep(sleep_time)

        # 流式任务结束后：关闭视频 writer，刷新 UI，切回参考图
        state.is_running = False
        
        # 关闭视频 writer
        if video_writer is not None:
            try:
                video_writer.close()

                # 确保有缩略图：如果没有保存首帧，尝试从视频提取或使用参考图
                if not first_frame_saved and video_path and Path(video_path).exists():
                    try:
                        reader = imageio.get_reader(video_path)
                        first_frame = reader.get_data(0)
                        reader.close()
                        img = Image.fromarray(first_frame)
                        img.save(thumbnail_path)
                    except Exception:
                        # 使用参考图作为缩略图
                        try:
                            ref_img_path = state.ref_image_path or DEFAULT_REF_IMAGE
                            if Path(ref_img_path).exists():
                                import shutil
                                shutil.copy(ref_img_path, thumbnail_path)
                        except Exception:
                            pass
            except Exception:
                pass
        
        state.current_video_writer = None
        state.current_video_path = None
        state.current_thumbnail_path = None
        
        # 使缓存失效，下次获取时会扫描目录
        state.invalidate_video_cache()
        
        final_ref = state.ref_image_path or DEFAULT_REF_IMAGE
        yield (
            final_ref,
            "**生成已停止。**",
            format_current_prompt_md(state.shared_prompt),
            _btn_update(),
            gr.update(visible=False),
            "",
        )

    except Exception as e:
        state.is_running = False
        # 异常时也尝试关闭视频 writer
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass
        state.current_video_writer = None
        state.current_video_path = None
        state.current_thumbnail_path = None
        state.invalidate_video_cache()
        yield (
            gr.skip(),
            f"**错误**：{e}",
            format_current_prompt_md(state.shared_prompt),
            _btn_update(),
            gr.update(visible=False),
            "",
        )
    finally:
        state.is_running = False
        state.played_frame_count = 0
        worker.join(timeout=WORKER_JOIN_TIMEOUT)
        try:
            pipeline.reset_stream(batch_size=1, dtype=torch.bfloat16, device=device, initial_latent=None)
        except Exception:
            pass
        try:
            if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()
            elif hasattr(pipeline.vae, "taehv") and hasattr(pipeline.vae.taehv, "reset"):
                pipeline.vae.taehv.reset()
        except Exception:
            pass
        state.is_stopping = False


# ── Gallery 刷新 ─────────────────────────────────────────────────────────────

def refresh_video_gallery():
    """刷新视频 Gallery，供生成结束和停止时复用。"""
    entries = state.get_video_gallery_entries()
    return gr.update(
        value=entries,
        columns=max(1, min(5, len(entries)))
    )


def show_completion_toast():
    """生成完成后显示 Toast 提示，供 .then() 链调用。"""
    if state.completion_status == "completed":
        gr.Info("✅ 视频生成完成", duration=20)
    # 重置状态，避免重复显示
    state.completion_status = None


# ── Gradio 回调 ───────────────────────────────────────────────────────────────

def check_model_ready_ui():
    """供 Timer 轮询：返回当前状态文案、按钮状态与 overlay 可见性。"""
    if state.model_ready:
        if DEBUG_FRONTEND_ONLY:
            return "**[仅前端调试]** 未加载模型，界面可正常操作。", _btn_update(), gr.skip()
        if state.is_running:
            # 推理中由 start_stream yield 推送帧率等详细信息，Timer 不覆盖
            return gr.skip(), _btn_update(), gr.skip()
        return "**就绪** — 请输入 Prompt 并点击「唤醒你的世界」。", _btn_update(), gr.update(visible=False)
    return "**模型加载中…** 加载完成后「唤醒你的世界」将可用。", _btn_update(), gr.skip()


def on_click_update_prompt(prompt: str):
    """按钮回调：根据是否已在生成，仅更新状态或启动流。
    参考图使用 state.ref_image_path（由右侧参考图列表选中更新）。
    异常仅通过 status 文案展示，不弹窗。
    """
    pm = format_current_prompt_md((prompt or "").strip())
    if DEBUG_FRONTEND_ONLY:
        yield gr.skip(), "**[仅前端调试]** 未加载模型，点击无实际效果。", pm, _btn_update(), gr.skip(), gr.skip()
        return
    if not state.model_ready:
        yield gr.skip(), "**模型加载中…** 请稍候再试。", pm, _btn_update(), gr.skip(), gr.skip()
        return
    ref_path = state.ref_image_path
    if state.is_running:
        status_md, prompt_md, btn_state = update_state(prompt, ref_path)
        yield gr.skip(), status_md, prompt_md, btn_state, gr.skip(), gr.skip()
        return
    try:
        # 启动前先显示 overlay 标签（"实时视频画面"），首帧到达后由 start_stream 隐藏
        yield gr.skip(), "**准备中…**", pm, _btn_update(), gr.update(visible=True), gr.skip()
        yield from start_stream(prompt, ref_path)
    except Exception as e:
        state.is_running = False
        yield gr.skip(), f"**错误**：{e}", format_current_prompt_md(state.shared_prompt), _btn_update(), gr.update(visible=False), ""


def on_stop():
    """「封存你的世界」回调。"""
    was_running = state.is_running
    state.is_running = False
    state.waiting_for_new_frames = False
    state.played_frame_count = 0
    if was_running:
        state.is_stopping = True
    final_ref = state.ref_image_path or DEFAULT_REF_IMAGE
    return (
        final_ref,
        "**正在停止...**",
        format_current_prompt_md(state.shared_prompt),
        _btn_update(),
        gr.update(visible=False),
        "",
    )
