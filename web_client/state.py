"""
web_client/state.py — 全局共享状态（单例）。
"""
import os
import time
import threading
from pathlib import Path

from web_client.config import MAX_VIDEO_HISTORY, AUTO_DELETE_OLD_VIDEOS, OUTPUT_DIR
from web_client.example_scenes import ExampleSkill


class StreamState:
    def __init__(self):
        self.shared_prompt: str = ""
        self.is_running: bool = False
        self.waiting_for_new_frames: bool = False
        self.block_count: int = 0
        self.frame_count: int = 0
        self.start_time: float = 0.0
        self.ref_image_path: str | None = None
        self.example_skill: ExampleSkill | None = None
        # 键盘交互状态
        self.frontend_pressed: set = set()    # 当前物理按住的键
        self.frontend_activated: set = set()  # 自上次上报起曾激活的键（未被 worker 消费）
        self.key_snapshot: dict = {}           # 最新一次 block 采样的按键状态（大写键名 -> bool）
        self._key_lock = threading.Lock()     # 保护 activated 的读写锁
        self.worker_error: str | None = None  # worker 异常时写入，start_stream 通过 status 展示后清空
        self.model_ready: bool = False  # 后台模型加载完成后为 True，「重塑你的世界」方可点击
        self.is_stopping: bool = False  # 停止中过渡态：on_stop 设置，start_stream finally 清除
        self.completion_status: str | None = None  # 完成状态: "completed" | "stopped" | None
        self.completion_deadline: float | None = None  # 完成提示倒计时结束时刻（time.monotonic）
        self.played_frame_count: int = 0  # 已经播放（yield）的帧数
        # 视频生成状态（仅用于当前正在生成的视频）
        self.current_video_writer = None       # 当前正在写入的 video writer
        self.current_video_path: str | None = None  # 当前视频文件路径
        self.current_thumbnail_path: str | None = None  # 当前缩略图文件路径
        self._video_cache: list[tuple[str, str, float]] | None = None  # (视频路径, 缩略图路径, mtime) 缓存
        self._cache_time: float = 0.0  # 缓存更新时间

    def reset_stats(self):
        self.block_count = 0
        self.frame_count = 0
        self.played_frame_count = 0
        self.start_time = time.time()
        self.completion_status = None
        self.completion_deadline = None

    def get_status_data(self, current_prompt: str) -> dict:
        """Return raw status data without any HTML formatting."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            "prompt": current_prompt or "",
            "block_count": self.block_count,
            "frame_count": self.frame_count,
            "elapsed": elapsed,
            "bps": self.block_count / elapsed if elapsed > 0 else 0,
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
        }

    def format_status(self, current_prompt: str) -> str:
        """Deprecated: use get_status_data() + format_status_md() instead."""
        from ui_helpers import format_status_md
        return format_status_md(self.get_status_data(current_prompt))

    def _scan_outputs_dir(self) -> list[tuple[str, str, float]]:
        """扫描 outputs 目录，返回 [(视频路径, 缩略图路径, mtime), ...]。
        
        按修改时间降序排列（最新的在前）。
        """
        videos = []
        if not OUTPUT_DIR.exists():
            return videos
        
        # 查找所有视频文件
        for video_file in OUTPUT_DIR.glob("stream_*.mp4"):
            # 查找对应的缩略图
            thumbnail_file = video_file.with_suffix(".png")
            if not thumbnail_file.exists():
                continue
            
            try:
                mtime = video_file.stat().st_mtime
                videos.append((str(video_file), str(thumbnail_file), mtime))
            except OSError:
                continue
        
        # 按修改时间降序排列
        videos.sort(key=lambda x: x[2], reverse=True)
        return videos

    def _cleanup_old_videos(self, videos: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        """清理超出限制的旧视频。
        
        Returns:
            清理后的视频列表
        """
        if len(videos) <= MAX_VIDEO_HISTORY:
            return videos
        
        # 需要删除的视频（最旧的）
        to_remove = videos[MAX_VIDEO_HISTORY:]
        for video_path, thumbnail_path, _ in to_remove:
            if AUTO_DELETE_OLD_VIDEOS:
                try:
                    os.remove(video_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                try:
                    os.remove(thumbnail_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
        
        return videos[:MAX_VIDEO_HISTORY]

    def get_video_gallery_entries(self) -> list[tuple[str, str]]:
        """获取 Videos Gallery 所需的条目列表。
        
        扫描 outputs 目录，返回 [(缩略图路径, 视频路径), ...]。
        使用缓存避免频繁 IO，缓存有效期 1 秒。
        """
        now = time.time()
        
        # 缓存有效，直接返回
        if self._video_cache is not None and (now - self._cache_time) < 1.0:
            return [(thumb, video) for video, thumb, _ in self._video_cache]
        
        # 扫描目录
        videos = self._scan_outputs_dir()
        
        # 清理旧视频
        videos = self._cleanup_old_videos(videos)
        
        # 更新缓存
        self._video_cache = videos
        self._cache_time = now

        return [(thumb, video) for video, thumb, _ in videos]

    def invalidate_video_cache(self):
        """使视频缓存失效，下次获取时会重新扫描目录。"""
        self._video_cache = None
        self._cache_time = 0.0


# 全局单例
state = StreamState()
