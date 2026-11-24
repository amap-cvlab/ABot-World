"""
ABot-World - 实时可交互世界模型 Gradio UI

入口文件：定义 Gradio UI 布局、事件绑定、服务启动。
业务逻辑由 config / state / keyboard / inference / pipeline_loader 模块提供。
"""

# ── A. 环境引导（必须在所有项目导入之前执行）────────────────────────────────
import sys
import os
import time
import threading
from functools import partial
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)
os.environ["PROJECT_ROOT"] = str(_project_root)

_gradio_tmp = _project_root / ".gradio_cache"
_gradio_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_gradio_tmp)

# ── B. 导入 ─────────────────────────────────────────────────────────────────
import gradio as gr

from web_client.config import (
    WEB_DIR, DEBUG_FRONTEND_ONLY, STREAM_HEIGHT,
    SERVER_NAME, SERVER_PORT, SHUTDOWN_GRACE_SECONDS,
    VAE_TYPE,
    USE_FP8_GEMM,
)
from web_client.state import state
from web_client.example_scenes import DEFAULT_EXAMPLE, EXAMPLE_SCENES, SKILLS_PER_EXAMPLE
from web_client.keyboard import on_key_update
from web_client.inference import (
    on_click_update_prompt, on_stop, check_model_ready_ui, show_completion_toast,
    on_example_skill,
)
from web_client.ui_helpers import format_current_prompt_md
from abot_world.iris.interactive_loader import get_pipeline, _init_lock

# ── C. 前端资源加载 ─────────────────────────────────────────────────────────
_KEY_HUD_WASD_HTML = (WEB_DIR / "key_hud_wasd.html").read_text(encoding="utf-8")
_KEY_HUD_IJKL_HTML = (WEB_DIR / "key_hud_ijkl.html").read_text(encoding="utf-8")
_KEY_HANDLER_JS = (WEB_DIR / "key_handler.js").read_text(encoding="utf-8")

# ── D. 主题 CSS（从外部文件加载）─────────────────────────────────────────
_theme_path = WEB_DIR / "theme.css"
_THEME_CSS = _theme_path.read_text(encoding="utf-8") if _theme_path.is_file() else ""
_HUD_CSS = (WEB_DIR / "key_hud.css").read_text(encoding="utf-8")
_PROMPT_BTNS_CSS = (WEB_DIR / "prompt_buttons.css").read_text(encoding="utf-8")
_COMBINED_CSS = _THEME_CSS + "\n" + _HUD_CSS + "\n" + _PROMPT_BTNS_CSS

# ── E. 示例画廊 ───────────────────────────────────────────────────────────
GALLERY_EXAMPLES = tuple(EXAMPLE_SCENES.values())
REF_IMAGE_ENTRIES = [(str(example.first_frame), example.label) for example in GALLERY_EXAMPLES]
INITIAL_PROMPT = DEFAULT_EXAMPLE.scene_prompt
state.ref_image_path = str(DEFAULT_EXAMPLE.first_frame)


def on_ref_gallery_select(evt: gr.SelectData):
    """画廊选中：
    1) 更新首帧参考图路径（影响后续生成/流结束后的最终替换）。
    2) 若正在流式生成：请求停止，让已入队帧先播放完，start_stream 最后会替换为选中图。
    3) 若未在生成：立刻用选中图替换 gr.Image。
    """
    if not isinstance(evt.index, int) or not 0 <= evt.index < len(GALLERY_EXAMPLES):
        return gr.skip(), gr.skip(), gr.skip(), *(gr.skip() for _ in range(SKILLS_PER_EXAMPLE))

    example = GALLERY_EXAMPLES[evt.index]
    path = str(example.first_frame)
    state.ref_image_path = path
    skill_buttons = [gr.update(value=skill.name) for skill in example.skills]

    if state.is_running:
        state.is_stopping = True
        state.is_running = False
        state.waiting_for_new_frames = False
        return example.scene_prompt, gr.skip(), example.image_hash, *skill_buttons

    return example.scene_prompt, gr.update(value=path), example.image_hash, *skill_buttons


def on_video_gallery_select(evt: gr.SelectData):
    """Videos Gallery 选中回调：显示视频播放区并播放选中的视频。
    
    Gallery 条目格式为 (缩略图路径, 视频路径)，evt.value 结构为:
    {'image': {...}, 'caption': '视频路径'}
    """
    try:
        # evt.value 是字典，视频路径在 caption 字段
        video_path = None
        if isinstance(evt.value, dict):
            video_path = evt.value.get('caption')
        elif isinstance(evt.value, str):
            video_path = evt.value

        if video_path and Path(video_path).exists():
            return gr.update(value=video_path), gr.update(visible=True)
        return gr.skip(), gr.skip()
    except Exception:
        return gr.skip(), gr.skip()


def on_close_video_player():
    """关闭视频播放区。"""
    return gr.update(value=None), gr.update(visible=False)


# ── F. Gradio Blocks UI 定义 ────────────────────────────────────────────────
with gr.Blocks(title="ABot-World - 实时可交互世界模型") as demo:
    gr.Markdown(
        "# ABot-World - 实时可交互世界模型\n\n",
        elem_classes=["fe-header", "fe-header--intro-gap"],
    )
    # --- 键盘交互：隐藏 Textbox 供 JS 上报按键 ---
    # Gradio：visible=False 时整块不挂载到 DOM，key_handler 无法找到 #key-state-input。
    # visible="hidden" 不占版面但仍存在于 DOM，可与 .key-state-hidden 配合完全隐藏。
    key_state_input = gr.Textbox(
        value="",
        elem_id="key-state-input",
        visible="hidden",
        label=None,
        elem_classes=["key-state-hidden"],
    )
    # ── 主区：实时画面 ────────────────────────────────────────────────────────
    with gr.Column(elem_classes=["fe-video-wrap"]):
        overlay_label = gr.HTML(
            '<div class="fe-video-overlay-label">正在链接你的世界</div>',
            visible=not DEBUG_FRONTEND_ONLY,
            elem_classes=["fe-overlay-wrapper"],
        )
        # 初始无图：去掉所有填充图标
        image_output = gr.Image(
            label=None,
            value=state.ref_image_path,
            visible=True,
            height=STREAM_HEIGHT,
            show_label=False,
            elem_classes=["fe-video-fixed"],
            format="webp"
        )

        # ── 进度条：frame 生成与播放进度 ──────────────────────────────────────────
        progress_bar = gr.HTML(
            value="",
            elem_classes=["fe-progress-container"],
            visible=True,
        )

        gr.Markdown(
            "**基础动作** · `Ctrl` 蹲下、`Shift` 跑步、`Space` 跳跃、`Alt` 翻滚 · "
            "画面内按住右键瞄准、点击左键普攻"
        )

        # ── 控制区：一行 左|中|右（模型状态与当前 Prompt 见页面底部 Debug 区，Ctrl+D 切换显示）──
        with gr.Column(elem_classes=["fe-controls-stack"]):
            with gr.Row(elem_classes=["fe-bottom-row"]):
                with gr.Column(scale=9, min_width=132, elem_classes=["fe-hud-col", "fe-hud-cyber"]):
                    key_display_wasd = gr.HTML(_KEY_HUD_WASD_HTML)

                with gr.Column(scale=30, elem_classes=["fe-prompt-col"]):
                    prompt_input = gr.Textbox(
                        label="描绘你的世界",
                        lines=5,
                        placeholder="e.g., A cat walking on a sunny beach...",
                        value=INITIAL_PROMPT,
                    )
                    with gr.Row(elem_classes=["fe-prompt-btns"]):
                        update_btn = gr.Button("唤醒你的世界", variant="primary", interactive=DEBUG_FRONTEND_ONLY)
                        stop_btn = gr.Button("封存你的世界", variant="stop")
                    example_selection = gr.State(DEFAULT_EXAMPLE.image_hash)
                    with gr.Row():
                        example_skill_buttons = [
                            gr.Button(skill.name) for skill in DEFAULT_EXAMPLE.skills
                        ]

                with gr.Column(scale=9, min_width=132, elem_classes=["fe-hud-col", "fe-hud-cyber"]):
                    key_display_ijkl = gr.HTML(_KEY_HUD_IJKL_HTML)

    # ── 场景预设（探索平行宇宙）────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**探索平行宇宙**", elem_classes=["fe-ref-title"])
            ref_gallery = gr.Gallery(
                value=REF_IMAGE_ENTRIES,
                label=None,
                columns=None,  # 自适应单行横向排列
                rows=1,
                height=172,
                object_fit="cover",
                show_label=False,
                allow_preview=False,
                interactive=False,
                elem_classes=["fe-ref-gallery-hscroll"],
            )

    # ── Debug 区：默认隐藏，前端 Ctrl+D 切换（见 key_handler.js / theme .fe-debug-panel）──
    with gr.Column(elem_classes=["fe-debug-panel"]):
        with gr.Row(elem_classes=["fe-meta-below-row", "fe-debug-panel__row"]):
            with gr.Column(scale=1, min_width=0, elem_classes=["fe-meta-slot-3-wrap"]):
                status_output = gr.Markdown(
                    value="**[仅前端调试]** 未加载模型，界面可正常操作。" if DEBUG_FRONTEND_ONLY else "**模型加载中…** 加载完成后「唤醒你的世界」将可用。",
                    elem_classes=["fe-status", "fe-status-inline", "fe-meta-slot-3"],
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["fe-meta-slot-4-wrap", "fe-vae-hud-wrap"]):
                current_prompt_display = gr.Markdown(
                    value=format_current_prompt_md(INITIAL_PROMPT),
                    elem_classes=["fe-current-prompt", "fe-status-inline"],
                )

    model_ready_timer = gr.Timer(2)
    model_ready_timer.tick(
        fn=check_model_ready_ui,
        inputs=[],
        outputs=[status_output, update_btn, overlay_label],
    )

    # ── G. 事件绑定 ───────────────────────────────────────────────────────────
    stream_event = update_btn.click(
        fn=on_click_update_prompt,
        inputs=[prompt_input],
        outputs=[image_output, status_output, current_prompt_display, update_btn, overlay_label, progress_bar],
    ).then(
        fn=show_completion_toast,
        inputs=[],
        outputs=[],
    )

    ref_gallery.select(
        fn=on_ref_gallery_select,
        inputs=None,
        outputs=[prompt_input, image_output, example_selection, *example_skill_buttons],
    )

    for slot, button in enumerate(example_skill_buttons):
        button.click(fn=partial(on_example_skill, slot), inputs=[example_selection], queue=False)

    stop_btn.click(
        fn=on_stop,
        outputs=[image_output, status_output, current_prompt_display, update_btn, overlay_label, progress_bar],
        cancels=[stream_event],
    )

    key_state_input.change(
        fn=on_key_update,
        inputs=[key_state_input],
        outputs=[],
        concurrency_limit="default",
        concurrency_id="key_update",
    )

demo.queue(default_concurrency_limit=2)

# ── H. Main 入口 + 启动 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if DEBUG_FRONTEND_ONLY:
        state.model_ready = True
    else:
        def _load_pipeline_background():
            """后台加载模型，完成后设置 state.model_ready，Timer 轮询会启用「唤醒你的世界」。"""
            print("[INIT] Loading pipeline in background...")
            with _init_lock:
                get_pipeline(
                    vae_type=VAE_TYPE,
                    use_fp8_gemm=USE_FP8_GEMM,
                )
            state.model_ready = True
            print("[INIT] Pipeline ready. 「唤醒你的世界」已可用。")

        threading.Thread(target=_load_pipeline_background, daemon=True).start()
        print("[INIT] Web UI starting (model loads in background).")

    try:
        demo.launch(
            server_name=SERVER_NAME,
            server_port=SERVER_PORT,
            js=_KEY_HANDLER_JS,
            css=_COMBINED_CSS,
            footer_links=[],  # 隐藏 Gradio 默认页脚（API / Built with Gradio / 设置）
        )
    except KeyboardInterrupt:
        pass
    finally:
        state.is_running = False
        # 给 worker 线程最多 SHUTDOWN_GRACE_SECONDS 秒退出，超时后强制结束进程
        # （GPU 推理线程无法被 interrupt，os._exit 是唯一可靠的退出方式）
        def _force_exit():
            time.sleep(SHUTDOWN_GRACE_SECONDS)
            os._exit(0)
        threading.Thread(target=_force_exit, daemon=True).start()
