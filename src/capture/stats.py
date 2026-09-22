"""
捕获统计信息 - 各捕获后端共享

单独成模块，避免 bettercam_capture 反向依赖 mss_capture。
"""
from dataclasses import dataclass


@dataclass
class CaptureStats:
    """捕获统计信息"""
    fps: float = 0.0
    frame_count: int = 0
    last_capture_time: float = 0.0
